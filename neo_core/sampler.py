# Port of Forge Neo's sampling pipeline:
#   modules/sd_samplers_kdiffusion.py   (sampler table, get_sigmas, sample, sample_img2img)
#   modules/sd_samplers_timesteps.py    (DDIM / PLMS)
#   modules_forge/forge_alter_samplers.py (CFG++ samplers)
#   modules/sd_samplers_common.py       (initialize, TorchHijack, create_noise_sampler, setup_img2img_steps)
#   modules/sd_samplers_cfg_denoiser.py (CFGDenoiser.forward)
#   backend/sampling/sampling_function.py (sampling_function_inner)
# The only part delegated to ComfyUI is running the diffusion model on a batch of conds
# (comfy.samplers.calc_cond_batch) so that LoRA / ControlNet / model patches keep working.
import inspect
import logging
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import torch

from . import k_sampling, samplers_extra, schedulers, timesteps_impl
from .rng import randn_local

logger = logging.getLogger("NeoSampler")

OOM_ERRORS = (getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError), torch.cuda.OutOfMemoryError)


# region settings (Neo "Settings > Sampler parameters" defaults)


@dataclass
class NeoOptions:
    eta_ancestral: float = 1.0
    eta_ddim: float = 0.0
    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = 0.0
    s_noise: float = 1.0
    sigma_min: float = 0.0
    sigma_max: float = 0.0
    rho: float = 0.0
    always_discard_next_to_last_sigma: bool = False
    sgm_noise_multiplier: bool = False
    beta_dist_alpha: float = 0.6
    beta_dist_beta: float = 0.6
    skip_early_cond: float = 0.0
    s_min_uncond: float = 0.0
    s_min_uncond_all: bool = False
    img2img_extra_noise: float = 0.0
    img2img_fix_steps: bool = False
    use_dynamic_shifting: bool = False
    invert_sigmas: bool = False
    use_karras_sigmas: bool = False
    use_exponential_sigmas: bool = False
    use_beta_sigmas: bool = False


# endregion

# region sampler table


@dataclass
class SamplerConfig:
    name: str
    func: object
    options: dict
    kind: str = "kdiffusion"  # or "timesteps"

    def total_steps(self, steps):
        if self.options.get("second_order", False):
            steps = steps * 2
        return steps


_K = [
    ("DPM++ 2M", "sample_dpmpp_2m", {"scheduler": "karras"}),
    ("DPM++ SDE", "sample_dpmpp_sde", {"scheduler": "karras", "second_order": True, "brownian_noise": True}),
    ("DPM++ 2M SDE", "sample_dpmpp_2m_sde", {"scheduler": "exponential", "brownian_noise": True}),
    ("DPM++ 3M SDE", "sample_dpmpp_3m_sde", {"scheduler": "exponential", "discard_next_to_last_sigma": True, "brownian_noise": True}),
    ("DPM++ 2s a RF", "sample_dpmpp_2s_ancestral_RF", {}),
    ("Euler a", "sample_euler_ancestral", {"uses_ensd": True}),
    ("Euler", "sample_euler", {}),
    ("ER SDE", "sample_er_sde", {}),
    ("LCM", "sample_lcm", {}),
    ("LMS", "sample_lms", {}),
    ("Heun", "sample_heun", {"second_order": True}),
    ("DPM2", "sample_dpm_2", {"scheduler": "karras", "discard_next_to_last_sigma": True, "second_order": True}),
    ("Res Multistep", "sample_res_multistep", {}),
    ("Kohaku LoNyu Yog", "sample_Kohaku_LoNyu_Yog", {}),
    ("Restart", samplers_extra.restart_sampler, {"scheduler": "karras", "second_order": True}),
    ("UniPC", samplers_extra.sample_unipc, {"discard_next_to_last_sigma": True}),
]

sampler_extra_params = {
    "sample_dpmpp_sde": ["eta", "s_noise", "r"],
    "sample_dpmpp_2m_sde": ["eta", "s_noise"],
    "sample_dpmpp_3m_sde": ["eta", "s_noise"],
    "sample_euler_ancestral": ["eta", "s_noise"],
    "sample_euler": ["s_churn", "s_tmin", "s_tmax", "s_noise"],
    "sample_heun": ["s_churn", "s_tmin", "s_tmax", "s_noise"],
    "sample_dpm_2": ["s_churn", "s_tmin", "s_tmax", "s_noise"],
}

SAMPLERS: dict[str, SamplerConfig] = {}
for _label, _func, _opts in _K:
    SAMPLERS[_label] = SamplerConfig(_label, _func, _opts)
SAMPLERS["DDIM"] = SamplerConfig("DDIM", timesteps_impl.ddim, {}, kind="timesteps")
SAMPLERS["PLMS"] = SamplerConfig("PLMS", timesteps_impl.plms, {}, kind="timesteps")
for _label, _key in (("DPM++ 2M CFG++", "dpmpp_2m_cfg_pp"), ("Euler a CFG++", "euler_ancestral_cfg_pp"), ("Euler CFG++", "euler_cfg_pp")):
    _base = _label.removesuffix(" CFG++")
    _cfg = next((o for n, _, o in _K if n == _base), {})
    SAMPLERS[_label] = SamplerConfig(_label, getattr(k_sampling, f"sample_{_key}"), dict(_cfg), kind="cfgpp")

SAMPLER_NAMES = list(SAMPLERS.keys())
SCHEDULER_NAMES = [s.label for s in schedulers.all_schedulers]


# endregion

# region model adapters


class PredictorAdapter:
    """Presents ComfyUI's model_sampling with the API of Neo's backend/modules/k_prediction.py"""

    def __init__(self, model_sampling):
        import comfy.model_sampling as cms

        self.ms = model_sampling
        if isinstance(model_sampling, cms.CONST):
            self.prediction_type = "const"
        elif isinstance(model_sampling, cms.V_PREDICTION):
            self.prediction_type = "v_prediction"
        else:
            self.prediction_type = "epsilon"
        self.sigma_data = getattr(model_sampling, "sigma_data", 1.0)

    @property
    def sigmas(self):
        return self.ms.sigmas

    @property
    def shift(self):
        return getattr(self.ms, "shift", 1.0)

    def sigma_min(self):
        return self.ms.sigmas[0]

    def sigma_max(self):
        return self.ms.sigmas[-1]

    def timestep(self, sigma):
        return self.ms.timestep(sigma)

    def sigma(self, timestep):
        return self.ms.sigma(timestep)

    def percent_to_sigma(self, percent):
        return self.ms.percent_to_sigma(percent)

    # k_prediction.AbstractPrediction.noise_scaling (verbatim)
    def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
        sigma = sigma.view(sigma.shape[:1] + (1,) * (noise.ndim - 1))
        if self.prediction_type == "const":
            noise_scale = 1.0
            return sigma * (noise_scale * noise) + (1.0 - sigma) * latent_image
        else:
            if max_denoise:
                noise = noise * torch.sqrt(1.0 + sigma**2.0)
            else:
                noise = noise * sigma

            noise += latent_image
            return noise


class ScheduleLinker:
    """k_diffusion/external.py ForgeScheduleLinker"""

    def __init__(self, predictor, is_sdxl=False):
        self.predictor = predictor
        self.is_sdxl = is_sdxl
        self.inner_model = SimpleNamespace(alphas_cumprod=None)

    @property
    def sigmas(self):
        return self.predictor.sigmas

    @property
    def log_sigmas(self):
        return self.predictor.sigmas.log()

    @property
    def sigma_min(self):
        return self.predictor.sigma_min()

    @property
    def sigma_max(self):
        return self.predictor.sigma_max()

    def get_sigmas(self, n=None):
        if n is None:
            return k_sampling.append_zero(self.sigmas.flip(0))
        t_max = len(self.sigmas) - 1
        t = torch.linspace(t_max, 0, n, device=self.sigmas.device)
        return k_sampling.append_zero(self.t_to_sigma(t))

    def sigma_to_t(self, sigma, quantize=None):
        return self.predictor.timestep(sigma)

    def t_to_sigma(self, t):
        return self.predictor.sigma(t)


class TorchHijack:
    """sd_samplers_common.TorchHijack"""

    def __init__(self, rng):
        self.rng = rng

    def __getattr__(self, item):
        if item == "randn_like":
            return self.randn_like

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{item}'")

    def randn_like(self, x):
        return self.rng.next()


# endregion

# region conditioning schedule (the cond side of CFGDenoiser)


@dataclass
class CondPart:
    weight: float
    schedule: list  # [(end_at_step, guider_key)]


def pick(schedule, step):
    """prompt_parser.reconstruct_cond_batch target selection"""
    target_index = 0
    for current, (end_at_step, _) in enumerate(schedule):
        if step <= end_at_step:
            target_index = current
            break
    return schedule[target_index][1]


# endregion


class InterruptedException(BaseException):
    pass


class NeoCFGDenoiser:
    """sd_samplers_cfg_denoiser.CFGDenoiser + sampling_function (ComfyUI runs the UNet)"""

    def __init__(self, run: "NeoRun", guider, base_model_options, linker, timesteps_mode):
        self.run = run
        self.guider = guider
        self.base_model_options = base_model_options
        self.inner_model = linker
        self.step = 0
        self.steps = None
        self.total_steps = None
        self.classic_ddim_eps_estimation = timesteps_mode

    def _conds(self, parts):
        out = []
        for part in parts:
            key = pick(part.schedule, self.step)
            for c in self.guider.conds[key]:
                c = c.copy()
                if part.weight is not None:
                    c["strength"] = part.weight  # compile_weighted_conditions
                out.append(c)
        return out

    def __call__(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond, **kwargs):
        import comfy.model_management
        import comfy.samplers

        opts = self.run.opts

        if self.classic_ddim_eps_estimation:
            acd = self.inner_model.inner_model.alphas_cumprod
            fake_sigmas = ((1 - acd) / acd) ** 0.5
            real_sigma = fake_sigmas[sigma.round().long().clip(0, int(fake_sigmas.shape[0]))]
            real_sigma_data = 1.0
            x = x * (((real_sigma**2.0 + real_sigma_data**2.0) ** 0.5)[:, None, None, None])
            sigma = real_sigma

        cond_list = self._conds(cond)
        uncond_list = self._conds(uncond) if uncond is not None else None

        if 0 < self.step / self.total_steps <= opts.skip_early_cond:
            cond_scale = 1.0
        elif (self.step % 2 or opts.s_min_uncond_all) and (0 < sigma[0] < s_min_uncond):
            cond_scale = 1.0

        # utils.join_dicts(unet_patcher.model_options, extra_model_options)
        model_options = _join_dicts(self.base_model_options, kwargs.get("model_options", {}))

        denoised = self._sampling_function(x, sigma, uncond_list, cond_list, cond_scale, model_options)

        self.run.last_denoised = denoised
        self.step += 1

        if self.classic_ddim_eps_estimation:
            eps = (x - denoised) / sigma[:, None, None, None]
            return eps

        return denoised

    def _calc_cond_batch(self, model, conds, x, timestep, model_options):
        import comfy.model_management as mm
        import comfy.samplers

        run = self.run
        if run.tile_size is None:
            try:
                return comfy.samplers.calc_cond_batch(model, conds, x, timestep, model_options)
            except OOM_ERRORS:
                mm.soft_empty_cache()
                run.tile_size = 128  # latent pixels (= 1024 px)
                logger.warning("[Neo] 采样显存不足，自动切换为分块采样（tile 1024px，重叠 128px）")
        while True:
            try:
                return tiled_calc_cond_batch(model, conds, x, timestep, model_options, run.tile_size, max(8, run.tile_size // 8))
            except OOM_ERRORS:
                mm.soft_empty_cache()
                if run.tile_size <= 32:
                    raise
                run.tile_size //= 2
                logger.warning(f"[Neo] 分块采样仍显存不足，缩小到 {run.tile_size * 8}px")

    def _sampling_function(self, x, timestep, uncond, cond, cond_scale, model_options):
        import comfy.samplers

        model = self.guider.inner_model
        edit_strength = sum((item["strength"] if "strength" in item else 1) for item in cond)

        if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
            uncond_ = None
        else:
            uncond_ = uncond

        conds = [cond, uncond_]
        if "sampler_calc_cond_batch_function" in model_options:
            args = {"conds": conds, "input": x, "sigma": timestep, "model": model, "model_options": model_options}
            out = model_options["sampler_calc_cond_batch_function"](args)
        else:
            out = self._calc_cond_batch(model, conds, x, timestep, model_options)

        # ComfyUI-style pre-cfg hooks (installed by ComfyUI nodes)
        for fn in model_options.get("sampler_pre_cfg_function", []):
            args = {"conds": conds, "conds_out": out, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep, "model": model, "model_options": model_options}
            out = fn(args)

        cond_pred, uncond_pred = out[0], out[1]

        if "sampler_cfg_function" in model_options:
            args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep, "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options, "input_cond": cond, "input_uncond": uncond_}
            cfg_result = x - model_options["sampler_cfg_function"](args)
        elif not math.isclose(edit_strength, 1.0):
            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * edit_strength
        else:
            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

        for fn in model_options.get("sampler_post_cfg_function", []):
            args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "cond_scale": cond_scale, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred, "sigma": timestep, "model_options": model_options, "input": x}
            cfg_result = fn(args)

        return cfg_result


def _tile_ramp(length, overlap, at_start, at_end, device):
    r = torch.ones(length, device=device)
    ov = min(overlap, length)
    if ov > 0:
        fade = torch.linspace(1.0 / (ov + 1), 1.0, ov, device=device)
        if at_start:
            r[:ov] = torch.minimum(r[:ov], fade)
        if at_end:
            r[-ov:] = torch.minimum(r[-ov:], fade.flip(0))
    return r


def _tile_starts(size, tile, overlap):
    if size <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, size - tile, stride))
    starts.append(size - tile)
    return starts


def tiled_calc_cond_batch(model, conds, x, timestep, model_options, tile, overlap):
    """VRAM fallback (not part of Neo): evaluate the model on overlapping spatial tiles and blend them
    (MultiDiffusion style). Only used after a real out-of-memory error."""
    import comfy.samplers

    H, W = x.shape[-2], x.shape[-1]
    outs = None
    weight = torch.zeros((H, W), device=x.device)
    for ys in _tile_starts(H, tile, overlap):
        for xs in _tile_starts(W, tile, overlap):
            th, tw = min(tile, H), min(tile, W)
            xt = x[..., ys : ys + th, xs : xs + tw]
            ot = comfy.samplers.calc_cond_batch(model, conds, xt, timestep, model_options)
            wy = _tile_ramp(th, overlap, ys > 0, ys + th < H, x.device)
            wx = _tile_ramp(tw, overlap, xs > 0, xs + tw < W, x.device)
            w = wy[:, None] * wx[None, :]
            if outs is None:
                outs = [torch.zeros_like(x) for _ in ot]
            for o, t in zip(outs, ot):
                o[..., ys : ys + th, xs : xs + tw] += t * w
            weight[ys : ys + th, xs : xs + tw] += w
    return [o / weight for o in outs]


def _join_dicts(base_dict, update_dict):
    """backend/utils.py join_dicts"""
    if not update_dict:
        return (base_dict or {}).copy()

    result = (base_dict or {}).copy()

    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _join_dicts(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            result[key] = result[key] + value
        else:
            result[key] = value

    return result


@dataclass
class NeoRun:
    """Everything the sampler needs (Neo's `p` object, trimmed)."""

    config: SamplerConfig
    scheduler: str
    steps: int
    cfg_scale: float
    opts: NeoOptions
    rng: object  # ImageRNG
    seeds: list
    predictor: PredictorAdapter
    is_sdxl: bool
    is_legacy: bool
    width: int
    height: int
    cond_parts: list  # [CondPart]
    uncond_parts: list  # [CondPart] or None
    img2img: bool = False
    denoising_strength: float = 1.0
    hires_mode: bool = True
    callback: object = None
    extra_generation_params: dict = field(default_factory=dict)
    last_denoised: object = None
    tile_size: object = None  # set when the VRAM fallback kicks in

    # ---- sd_samplers_kdiffusion.KDiffusionSampler.get_sigmas ----
    def get_sigmas(self, linker, steps):
        opts = self.opts
        discard_next_to_last_sigma = self.config.options.get("discard_next_to_last_sigma", False)
        if opts.always_discard_next_to_last_sigma and not discard_next_to_last_sigma:
            discard_next_to_last_sigma = True

        steps += 1 if discard_next_to_last_sigma else 0

        scheduler_name = self.scheduler or "Automatic"
        if scheduler_name == "Automatic":
            scheduler_name = self.config.options.get("scheduler", None)

            if scheduler_name is None and not self.is_legacy:
                scheduler_name = "Normal"

        scheduler = schedulers.schedulers_map.get(scheduler_name)

        m_sigma_min, m_sigma_max = linker.sigmas[0].item(), linker.sigmas[-1].item()

        schedulers.CTX.opts = opts
        schedulers.CTX.is_sdxl = self.is_sdxl

        if scheduler is None or scheduler.function is None:
            sigmas = linker.get_sigmas(steps)
        else:
            sigmas_kwargs = {"sigma_min": m_sigma_min, "sigma_max": m_sigma_max}

            if opts.sigma_min != 0 and opts.sigma_min != m_sigma_min:
                sigmas_kwargs["sigma_min"] = opts.sigma_min
            if opts.sigma_max != 0 and opts.sigma_max != m_sigma_max:
                sigmas_kwargs["sigma_max"] = opts.sigma_max

            if scheduler.default_rho != -1 and opts.rho != 0 and opts.rho != scheduler.default_rho:
                sigmas_kwargs["rho"] = opts.rho

            if scheduler.need_inner_model:
                sigmas_kwargs["inner_model"] = linker

            if scheduler.label == "Flux2":
                sigmas_kwargs["width"] = self.width
                sigmas_kwargs["height"] = self.height

            sigmas = scheduler.function(n=steps, **sigmas_kwargs, device=torch.device("cpu"))

        if discard_next_to_last_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])

        return sigmas.cpu()

    # ---- sd_samplers_common.setup_img2img_steps ----
    def setup_img2img_steps(self, steps):
        if self.opts.img2img_fix_steps or self.hires_mode:
            requested_steps = steps
            steps = int(requested_steps / min(self.denoising_strength, 0.999)) if self.denoising_strength > 0 else 0
            t_enc = requested_steps - 1
        else:
            t_enc = int(min(self.denoising_strength, 0.999) * steps)

        return steps, t_enc

    # ---- sd_samplers_common.Sampler.initialize ----
    def initialize(self, func, eta_default_field):
        opts = self.opts
        eta = getattr(opts, eta_default_field)
        extra_params = sampler_extra_params.get(func if isinstance(func, str) else None, [])
        fn = getattr(k_sampling, func) if isinstance(func, str) else func
        signature = inspect.signature(fn).parameters

        p_values = {
            "eta": None,
            "s_churn": opts.s_churn,
            "s_tmin": opts.s_tmin,
            "s_tmax": opts.s_tmax or float("inf"),
            "s_noise": opts.s_noise,
        }
        kwargs = {}
        for param_name in extra_params:
            if param_name in p_values and param_name in signature:
                kwargs[param_name] = p_values[param_name]

        if "eta" in signature:
            kwargs["eta"] = eta

        return fn, kwargs

    def create_noise_sampler(self, x, sigmas):
        sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
        return k_sampling.BrownianTreeNoiseSampler(x, sigma_min, sigma_max, seed=list(self.seeds))

    # ---- the actual sampling, called from inside ComfyUI's guider (model loaded & patched) ----
    def execute(self, guider, extra_args, noise, latent_image, disable_pbar):
        import torchsde._brownian.brownian_interval as bi

        base_model_options = extra_args.get("model_options", {})
        timesteps_mode = self.config.kind == "timesteps"
        linker = ScheduleLinker(self.predictor, self.is_sdxl)
        denoiser = NeoCFGDenoiser(self, guider, base_model_options, linker, timesteps_mode)

        cfg = self.rng.cfg
        old_randn = bi._randn
        old_torch = k_sampling.torch

        def torchsde_randn(size, dtype, device, seed):
            return randn_local(cfg, seed, size).to(device=device, dtype=dtype)

        bi._randn = torchsde_randn
        k_sampling.torch = TorchHijack(self.rng)
        try:
            if timesteps_mode:
                return self._run_timesteps(denoiser, linker, noise, latent_image, disable_pbar)
            return self._run_kdiffusion(denoiser, linker, noise, latent_image, disable_pbar)
        finally:
            bi._randn = old_randn
            k_sampling.torch = old_torch

    def _callback(self, denoiser, total):
        import comfy.model_management

        def cb(d):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if self.callback is not None:
                self.callback(d["i"], d["denoised"], d["x"], total)

        return cb

    def _launch(self, denoiser, steps, fn):
        denoiser.steps = steps
        denoiser.total_steps = self.config.total_steps(steps)
        return fn()

    def _extra_args(self):
        return {
            "cond": self.cond_parts,
            "image_cond": None,
            "uncond": self.uncond_parts,
            "cond_scale": self.cfg_scale,
            "s_min_uncond": self.opts.s_min_uncond,
        }

    def _run_kdiffusion(self, denoiser, linker, noise, latent_image, disable_pbar):
        opts = self.opts
        predictor = self.predictor
        fn, extra_params_kwargs = self.initialize(self.config.func, "eta_ancestral")
        parameters = inspect.signature(fn).parameters
        steps = self.steps

        if not self.img2img:
            # KDiffusionSampler.sample
            x = noise
            sigmas = self.get_sigmas(linker, steps).to(x.device)
            x = predictor.noise_scaling(sigmas[0], x, torch.zeros_like(x), max_denoise=opts.sgm_noise_multiplier)

            if "n" in parameters:
                extra_params_kwargs["n"] = steps
            if "sigma_min" in parameters:
                extra_params_kwargs["sigma_min"] = linker.sigmas[0].item()
                extra_params_kwargs["sigma_max"] = linker.sigmas[-1].item()
            if "sigmas" in parameters:
                extra_params_kwargs["sigmas"] = sigmas
            if self.config.options.get("brownian_noise", False):
                extra_params_kwargs["noise_sampler"] = self.create_noise_sampler(x, sigmas)

            n_steps = steps
            start = x
        else:
            # KDiffusionSampler.sample_img2img
            x = latent_image
            steps, t_enc = self.setup_img2img_steps(steps)
            sigmas = self.get_sigmas(linker, steps).to(x.device)
            sigma_sched = sigmas[steps - t_enc - 1 :]

            x = x.to(noise)
            xi = predictor.noise_scaling(sigma_sched[0], noise, x, max_denoise=False)
            if opts.img2img_extra_noise > 0:
                xi += noise * opts.img2img_extra_noise

            if "sigma_min" in parameters:
                extra_params_kwargs["sigma_min"] = sigma_sched[-2]
            if "sigma_max" in parameters:
                extra_params_kwargs["sigma_max"] = sigma_sched[0]
            if "n" in parameters:
                extra_params_kwargs["n"] = len(sigma_sched) - 1
            if "sigma_sched" in parameters:
                extra_params_kwargs["sigma_sched"] = sigma_sched
            if "sigmas" in parameters:
                extra_params_kwargs["sigmas"] = sigma_sched
            if self.config.options.get("brownian_noise", False):
                extra_params_kwargs["noise_sampler"] = self.create_noise_sampler(x, sigmas)

            n_steps = t_enc + 1
            start = xi

        if self.config.kind == "cfgpp" and self.cfg_scale > 2.0:
            logger.warning("CFG between 1.0 ~ 2.0 is recommended when using CFG++ samplers")

        total = len(extra_params_kwargs.get("sigmas", [0, 0])) - 1
        return self._launch(
            denoiser,
            n_steps,
            lambda: fn(denoiser, start, extra_args=self._extra_args(), disable=disable_pbar, callback=self._callback(denoiser, total), **extra_params_kwargs),
        )

    def _run_timesteps(self, denoiser, linker, noise, latent_image, disable_pbar):
        # sd_samplers_timesteps.CompVisSampler
        opts = self.opts
        device = noise.device
        alphas_cumprod = (1.0 / (self.predictor.sigmas**2.0 + 1.0)).to(device)
        linker.inner_model.alphas_cumprod = alphas_cumprod
        fn, extra_params_kwargs = self.initialize(self.config.func, "eta_ddim")
        parameters = inspect.signature(fn).parameters

        def get_timesteps(steps):
            discard_next_to_last_sigma = self.config.options.get("discard_next_to_last_sigma", False)
            if opts.always_discard_next_to_last_sigma and not discard_next_to_last_sigma:
                discard_next_to_last_sigma = True
            steps += 1 if discard_next_to_last_sigma else 0
            return torch.clip(torch.asarray(list(range(0, 1000, 1000 // steps)), device=device) + 1, 0, 999)

        if not self.img2img:
            timesteps = get_timesteps(self.steps)
            if "timesteps" in parameters:
                extra_params_kwargs["timesteps"] = timesteps
            n_steps = self.steps
            start = noise
            total = len(timesteps) - 1
        else:
            x = latent_image
            steps, t_enc = self.setup_img2img_steps(self.steps)
            timesteps = get_timesteps(steps)
            timesteps_sched = timesteps[:t_enc]
            sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[timesteps[t_enc]])
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[timesteps[t_enc]])
            xi = x.to(noise) * sqrt_alpha_cumprod + noise * sqrt_one_minus_alpha_cumprod
            if opts.img2img_extra_noise > 0:
                xi += noise * opts.img2img_extra_noise * sqrt_alpha_cumprod
            if "timesteps" in parameters:
                extra_params_kwargs["timesteps"] = timesteps_sched
            if "is_img2img" in parameters:
                extra_params_kwargs["is_img2img"] = True
            n_steps = t_enc + 1
            start = xi
            total = max(len(timesteps_sched) - 1, 1)

        return self._launch(
            denoiser,
            n_steps,
            lambda: fn(denoiser, start, extra_args=self._extra_args(), disable=disable_pbar, callback=self._callback(denoiser, total), **extra_params_kwargs),
        )
