import copy
import logging
import os
import re

import torch

import comfy.model_base
import comfy.model_management
import comfy.sample
import comfy.samplers
import comfy.sd
import comfy.utils
import folder_paths
import latent_preview

from .neo_core import prompt_parser
from .neo_core.rng import ImageRNG, RNGConfig
from .neo_core.sampler import SAMPLER_NAMES, SAMPLERS, SCHEDULER_NAMES, CondPart, NeoOptions, NeoRun, PredictorAdapter, ScheduleLinker
from .neo_core.text_engines import EMPHASIS_MODES, build_engine, detect_engine_kind

logger = logging.getLogger("NeoSampler")

CATEGORY = "Neo (Forge Neo 复刻)"
INF = float("inf")

# modules/extra_networks.py
re_extra_net = re.compile(r"<(\w+):([^>]+)>")

# keys produced by text encoders that are replaced when Neo re-encodes a prompt
_ENCODER_KEYS = ("neo_prompt", "pooled_output", "t5xxl_ids", "t5xxl_weights", "attention_mask")

TE_PRECISIONS = {"fp16 (Neo 默认)": torch.float16, "bf16": torch.bfloat16, "fp32 (Comfy 默认)": torch.float32}

LATENT_TYPES = {
    "Anima / Wan / Qwen-Image (16通道, 视频式)": (16, 3),
    "SD1.5 / SDXL (4通道)": (4, 2),
    "Flux / SD3 / Z-Image / Lumina (16通道)": (16, 2),
}

# region CLIP wrapper


class NeoTokens(dict):
    """ComfyUI's normal token dict + the raw prompt text (so Neo can redo tokenization its own way)."""

    neo_text = None


class NeoPrompt:
    """Travels inside CONDITIONING. Holds the raw text + the Neo CLIP, so the Neo sampler can
    re-encode it per prompt-schedule step (Neo computes schedules from the step count)."""

    def __init__(self, clip, text):
        self.clip = clip
        self.text = text
        self.kind = clip.neo_kind
        self.settings = dict(clip.neo_settings)
        self._cache = {}

    def encode(self, texts):
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            engine = build_engine(self.clip, self.kind, self.settings)
            for t, r in zip(todo, engine.encode_texts(todo)):
                self._cache[t] = r
        return {t: self._cache[t] for t in texts}

    def rebind(self, clip):
        """Same prompt text, encoded by another (e.g. LoRA-patched) Neo CLIP."""
        return NeoPrompt(clip, self.text)

    def static(self):
        """Fallback tensor for ordinary samplers: the prompt as Neo would see it at the last step."""
        text = re.sub(re_extra_net, "", self.text)  # <lora:...> would otherwise be encoded as text
        sched = prompt_parser.get_learned_conditioning_prompt_schedules([text], 1)[0]
        text = sched[-1][1]
        return self.encode([text])[text]


class NeoCLIP(comfy.sd.CLIP):
    neo_settings = None
    neo_kind = None

    def clone(self, *args, **kwargs):
        n = super().clone(*args, **kwargs)
        n.__class__ = NeoCLIP
        n.neo_settings = dict(self.neo_settings)
        n.neo_kind = self.neo_kind
        return n

    def tokenize(self, text, return_word_ids=False, **kwargs):
        tokens = super().tokenize(text, return_word_ids, **kwargs)
        if self.neo_kind is None or return_word_ids or not isinstance(text, str):
            return tokens
        t = NeoTokens(tokens)
        t.neo_text = text
        return t

    def encode_from_tokens(self, tokens, return_pooled=False, return_dict=False):
        text = getattr(tokens, "neo_text", None)
        if text is None or self.neo_kind is None:
            return super().encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=return_dict)

        payload = NeoPrompt(self, text)
        cond, extra = payload.static()
        cond = cond.clone()
        cond.neo_prompt = payload  # survives nodes that only keep the tensor (return_pooled API)

        if return_dict:
            out = {"cond": cond, "pooled_output": extra.get("pooled_output")}
            for k, v in extra.items():
                out[k] = v
            out["neo_prompt"] = payload
            if hasattr(self, "add_hooks_to_dict"):
                self.add_hooks_to_dict(out)
            return out
        if return_pooled:
            return cond, extra.get("pooled_output")
        return cond


def make_neo_clip(clip, settings):
    n = clip.clone()
    n.__class__ = NeoCLIP
    n.neo_settings = settings
    n.neo_kind = detect_engine_kind(n)
    return n


# endregion

# region helpers


def _seeds(seed, subseed, subseed_strength, batch):
    # processing.py: p.all_seeds / p.all_subseeds
    seeds = [int(seed) + (x if subseed_strength == 0 else 0) for x in range(batch)]
    subseeds = [int(subseed) + x for x in range(batch)]
    return seeds, subseeds


def _rng_from_meta(meta, shape):
    cfg = RNGConfig(meta["source"], comfy.model_management.get_torch_device(), meta["ensd"])
    return ImageRNG(cfg, shape, meta["seeds"], subseeds=meta["subseeds"], subseed_strength=meta["subseed_strength"], seed_resize_from_h=meta["seed_resize_from_h"], seed_resize_from_w=meta["seed_resize_from_w"])


def _apply_shift(model, shift):
    ms = model.get_model_object("model_sampling")
    if not hasattr(ms, "shift"):
        logger.warning("[Neo] 该模型没有 shift 参数，已忽略 shift 设置")
        return model
    m = model.clone()
    new = copy.deepcopy(ms)
    if hasattr(new, "multiplier"):
        new.set_parameters(shift=shift, multiplier=new.multiplier)
    else:
        new.set_parameters(shift=shift)
    m.add_object_patch("model_sampling", new)
    return m


def _payload_of(entry):
    tensor, d = entry
    return d.get("neo_prompt", None) or getattr(tensor, "neo_prompt", None)


def _parse_prompt_loras(text):
    """modules/extra_networks.py parse_prompt + extensions-builtin/sd_forge_lora/extra_networks_lora.py activate"""
    out = []
    for m in re_extra_net.finditer(text):
        if m.group(1) != "lora":
            continue
        positional, named = [], {}
        for item in m.group(2).split(":"):
            parts = item.split("=", 2)
            if len(parts) == 2:
                named[parts[0]] = parts[1]
            else:
                positional.append(item)
        name = positional[0]
        if len(positional) > 1:
            te = 0.0 if "@" in positional[1] else float(positional[1])
        else:
            te = 1.0
        te = float(named.get("te", te))
        unet = float(positional[2]) if len(positional) > 2 else te
        unet = float(named.get("unet", unet))
        out.append((name, te, unet))
    return out


_lora_index = {"key": None, "names": {}, "aliases": {}}


def _lora_alias(path):
    if not path.lower().endswith(".safetensors"):
        return None
    try:
        import safetensors

        with safetensors.safe_open(path, framework="pt") as f:
            return (f.metadata() or {}).get("ss_output_name")
    except Exception:
        return None


def _find_lora(name):
    """sd_forge_lora networks.py: match by file name (without extension), then by ss_output_name alias"""
    files = folder_paths.get_filename_list("loras")
    key = tuple(files)
    idx = _lora_index
    if idx["key"] != key:
        idx["key"], idx["names"], idx["aliases"] = key, {}, {}
        for f in files:
            if os.path.splitext(f)[1].lower() not in (".safetensors", ".pt", ".ckpt"):
                continue
            idx["names"].setdefault(os.path.splitext(os.path.basename(f))[0], []).append(f)
    hits = idx["names"].get(name)
    if hits:
        if len(hits) > 1:
            logger.warning(f"[Neo] 有多个同名 LoRA「{name}」: {hits}，使用 {hits[-1]}")
        return hits[-1]
    if not idx["aliases"]:
        for f in files:
            alias = _lora_alias(folder_paths.get_full_path("loras", f))
            if alias:
                idx["aliases"][alias] = f
    return idx["aliases"].get(name)


_lora_cache = {}


def _load_lora_sd(rel):
    path = folder_paths.get_full_path("loras", rel)
    stamp = (path, os.path.getmtime(path))
    if stamp not in _lora_cache:
        _lora_cache.clear()
        _lora_cache[stamp] = comfy.utils.load_torch_file(path, safe_load=True)
    sd = _lora_cache[stamp]
    if any(k.startswith("lora_unet__") for k in sd):  # networks.py load_networks
        sd = {("lora_unet_" + k[len("lora_unet__"):] if k.startswith("lora_unet__") else k): v for k, v in sd.items()}
    return sd


def _apply_prompt_loras(model, positive, negative):
    """Neo: LoRAs named in the positive prompt are applied (in order) to the UNet and to the text encoder
    before both prompts are encoded. Uses comfy.sd.load_lora_for_models, i.e. exactly what LoraLoader does."""
    payloads = [p for p in (_payload_of(e) for e in positive) if p is not None]
    if not payloads:
        return model, {}
    tags = _parse_prompt_loras(payloads[0].text)
    if not tags:
        return model, {}
    if any(_parse_prompt_loras(p.text) for p in payloads[1:]):
        logger.warning("[Neo] 多个正面条件都写了 <lora>，与 Neo 相同只采用第一个提示词里的 LoRA")

    clips = {}
    for e in list(positive) + list(negative):
        p = _payload_of(e)
        if p is not None:
            clips[id(p.clip)] = p.clip
    if not clips:
        logger.warning("[Neo] 条件不是经 Neo CLIP 转换器编码的，<lora> 只能作用于模型，无法作用于文本编码器")

    new_clips = dict(clips)
    for name, te, unet in tags:
        rel = _find_lora(name)
        if rel is None:
            logger.error(f"[Neo] 找不到 LoRA「{name}」（与 Neo 相同：跳过）")
            continue
        sd = _load_lora_sd(rel)
        logger.info(f"[Neo] 提示词 LoRA: {rel}  unet={unet} te={te}")
        first = True
        for k in list(new_clips):
            if first:
                model, new_clips[k] = comfy.sd.load_lora_for_models(model, new_clips[k], sd, unet, te)
                first = False
            else:
                _, new_clips[k] = comfy.sd.load_lora_for_models(None, new_clips[k], sd, 0, te)
        if first:
            model, _ = comfy.sd.load_lora_for_models(model, None, sd, unet, 0)
    return model, new_clips


class _CondBuilder:
    def __init__(self, clip_map=None):
        self.conds = {}
        self._keys = {}
        self.clip_map = clip_map or {}
        self._rebound = {}

    def key(self, conditioning_entry, ident=None):
        if ident is not None and ident in self._keys:
            return self._keys[ident]
        k = f"neo_{len(self.conds)}"
        self.conds[k] = [conditioning_entry]
        if ident is not None:
            self._keys[ident] = k
        return k

    def expand(self, conditioning, is_positive, base_steps, hires_steps):
        parts = []
        for tensor, d in conditioning:
            payload = d.get("neo_prompt", None) or getattr(tensor, "neo_prompt", None)
            if payload is not None and id(getattr(payload, "clip", None)) in self.clip_map:
                if id(payload) not in self._rebound:
                    self._rebound[id(payload)] = payload.rebind(self.clip_map[id(payload.clip)])
                payload = self._rebound[id(payload)]
            if payload is None:
                k = self.key([tensor, d])
                parts.append(CondPart(weight=None, schedule=[(INF, k)]))
                continue

            text = payload.text
            base = {k: v for k, v in d.items() if k not in _ENCODER_KEYS}

            if is_positive:
                # StableDiffusionProcessing.parse_extra_network_prompts (positive prompts only)
                text = re.sub(re_extra_net, "", text)
                res_indexes, flat, _ = prompt_parser.get_multicond_prompt_list([text])
                subprompts, weights = list(flat), res_indexes[0]
            else:
                subprompts, weights = [text], [(0, None)]

            schedules = prompt_parser.get_learned_conditioning_prompt_schedules(subprompts, base_steps, hires_steps)
            encoded = payload.encode([t for sched in schedules for _, t in sched])

            for index, weight in weights:
                sched = []
                for end_at_step, t in schedules[index]:
                    z, extra = encoded[t]
                    dd = dict(base)
                    dd.update(extra)
                    k = self.key([z, dd], ident=(id(payload), t))
                    sched.append((end_at_step, k))
                parts.append(CondPart(weight=weight, schedule=sched))
        return parts


class _Bridge(comfy.samplers.Sampler):
    """Plugged into ComfyUI's CFGGuider: the model is loaded/patched and conds are processed by
    ComfyUI, then the whole Neo sampling loop runs here."""

    def __init__(self, run):
        self.run = run

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        self.run.callback = callback
        return self.run.execute(model_wrap, extra_args, noise, latent_image, disable_pbar)


# endregion

# region nodes


class NeoCLIPConverter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "emphasis": (EMPHASIS_MODES, {"default": "Original", "tooltip": "Neo 设置 → 强调模式 (Emphasis Mode)"}),
                "clip_skip": ("INT", {"default": 2, "min": 1, "max": 12, "tooltip": "Neo 设置 → Clip Skip（SD1.5/SDXL 有效；SDXL 最小按 2 处理，与 Neo 相同）"}),
                "anima_te_precision": (list(TE_PRECISIONS.keys()), {"default": "fp16 (Neo 默认)", "tooltip": "Anima 的 Qwen3 文本编码器计算精度。Neo 默认以 fp16 运行"}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "convert"
    CATEGORY = CATEGORY
    DESCRIPTION = "接在 CLIP 加载器 / LoRA 之后、任意文本编码节点之前。让文本编码按 Forge Neo 的规则进行（分词、权重、强调、调度语法等）。"

    def convert(self, clip, emphasis, clip_skip, anima_te_precision):
        settings = {"emphasis": emphasis, "clip_skip": int(clip_skip), "anima_te_dtype": TE_PRECISIONS[anima_te_precision]}
        n = make_neo_clip(clip, settings)
        if n.neo_kind is None:
            logger.warning("[Neo] 这个文本编码器暂不支持 Neo 规则（目前支持 Anima / SD1.5 / SDXL），将按 ComfyUI 原逻辑编码")
        else:
            logger.info(f"[Neo] CLIP 转换器：识别为 {n.neo_kind}")
        return (n,)


class NeoEmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 16384, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True}),
                "rng_source": (["GPU", "CPU", "NV"], {"default": "GPU", "tooltip": "Neo 设置 → 随机数生成器。注意 Neo 的出厂默认是 CPU"}),
                "latent_type": (list(LATENT_TYPES.keys()),),
                "eta_noise_seed_delta": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "tooltip": "Neo 设置 → ENSD"}),
            },
            "optional": {
                "model": ("MODEL", {"tooltip": "接上后按模型自动决定通道数/维度，忽略 latent_type"}),
                "latent": ("LATENT", {"tooltip": "图生图：接 VAE 编码后的 latent，为它配上 Neo 噪声"}),
                "subseed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "tooltip": "变异种子 (Variation seed)"}),
                "subseed_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed_resize_from_w": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "seed_resize_from_h": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY
    DESCRIPTION = "按 Forge Neo 的规则（ImageRNG）生成初始噪声，并记录随机数发生器状态，供 Neo K采样器 继续使用同一条随机序列。"

    def generate(self, width, height, batch_size, seed, rng_source, latent_type, eta_noise_seed_delta, model=None, latent=None, subseed=0, subseed_strength=0.0, seed_resize_from_w=0, seed_resize_from_h=0):
        if latent is not None:
            samples = latent["samples"]
            mode = "img2img"
            out = latent.copy()
        else:
            if model is not None:
                fmt = model.get_model_object("latent_format")
                channels = fmt.latent_channels
                dims = getattr(fmt, "latent_dimensions", 2)
                ratio = getattr(fmt, "spacial_downscale_ratio", 8)
            else:
                channels, dims = LATENT_TYPES[latent_type]
                ratio = 8
            if dims == 3:
                shape = [batch_size, channels, 1, height // ratio, width // ratio]
            else:
                shape = [batch_size, channels, height // ratio, width // ratio]
            samples = torch.zeros(shape, device=comfy.model_management.intermediate_device())
            mode = "txt2img"
            out = {}

        batch = samples.shape[0]
        seeds, subseeds = _seeds(seed, subseed, subseed_strength, batch)
        meta = {
            "source": rng_source,
            "ensd": int(eta_noise_seed_delta),
            "seeds": seeds,
            "subseeds": subseeds,
            "subseed_strength": float(subseed_strength),
            "seed_resize_from_h": int(seed_resize_from_h),
            "seed_resize_from_w": int(seed_resize_from_w),
            "mode": mode,
        }

        rng = _rng_from_meta(meta, tuple(samples.shape[1:]))
        noise = rng.next()  # the first noise, exactly like processing.py: x = self.rng.next()
        meta["shape"] = tuple(samples.shape[1:])
        meta["batch"] = batch
        meta["state"] = rng.get_state()
        meta["noise"] = noise.cpu()

        out["samples"] = samples
        out["neo_rng"] = meta
        return (out,)


class NeoKSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT", {"tooltip": "必须来自 Neo 空Latent（或它的下游，例如放大后的 latent）"}),
                "sampler_name": (SAMPLER_NAMES, {"default": "Euler a"}),
                "scheduler": (SCHEDULER_NAMES, {"default": "Automatic"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "只对图生图/高清修复生效（文生图时 Neo 没有重绘幅度）"}),
                "denoise_mode": (["Hires.fix（精确步数）", "图生图（步数×重绘幅度）"], {"tooltip": "Neo 的高清修复第二阶段会精确执行设定步数；图生图页面默认只执行 步数×重绘幅度 步"}),
                "shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.05, "tooltip": "Neo 的 Shift 滑条。0 = 使用模型自带值（Anima 为 3.0）"}),
            },
            "optional": {
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Eta for k-diffusion samplers"}),
                "eta_ddim": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "s_churn": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "s_tmin": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "s_tmax": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 999.0, "step": 0.01}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.1, "step": 0.001}),
                "sigma_min": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "sigma_max": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1000.0, "step": 0.001}),
                "rho": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "always_discard_penultimate_sigma": ("BOOLEAN", {"default": False}),
                "sgm_noise_multiplier": ("BOOLEAN", {"default": False}),
                "beta_alpha": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 2.0, "step": 0.01}),
                "beta_beta": ("FLOAT", {"default": 0.6, "min": 0.01, "max": 2.0, "step": 0.01}),
                "skip_early_cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Ignore Negative Prompt during Early Steps"}),
                "ngms": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8.0, "step": 0.05, "tooltip": "Skip Negative Prompt during Later Steps (sigma)"}),
                "ngms_all_steps": ("BOOLEAN", {"default": False}),
                "extra_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Extra Noise Multiplier for img2img and Hires. fix"}),
                "turbo": ("NEO_TURBO", {"tooltip": "接「Neo Turbo 加速」。接上并启用时覆盖采样器/调度器/步数/CFG"}),
                "prompt_lora": ("BOOLEAN", {"default": True, "label_on": "启用 <lora:…>", "label_off": "关闭", "tooltip": "按 Neo 规则加载正面提示词里的 <lora:名称:权重>（同时作用于模型和文本编码器）。已经用 LoRA 加载器加载的话请关闭，否则会叠加两次"}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY
    DESCRIPTION = "Forge Neo 采样器/调度器与采样流程的复刻。"

    def sample(self, model, positive, negative, latent_image, sampler_name, scheduler, steps, cfg, denoise, denoise_mode, shift,
               eta=1.0, eta_ddim=0.0, s_churn=0.0, s_tmin=0.0, s_tmax=0.0, s_noise=1.0, sigma_min=0.0, sigma_max=0.0, rho=0.0,
               always_discard_penultimate_sigma=False, sgm_noise_multiplier=False, beta_alpha=0.6, beta_beta=0.6,
               skip_early_cfg=0.0, ngms=0.0, ngms_all_steps=False, extra_noise=0.0, prompt_lora=True, turbo=None):
        meta = latent_image.get("neo_rng")
        if meta is None:
            raise ValueError("Neo K采样器：latent 必须来自「Neo 空Latent」（图生图时把 VAE 编码结果接到 Neo 空Latent 的 latent 输入）")

        if "noise_mask" in latent_image:
            logger.warning("[Neo] 暂不支持局部重绘遮罩，noise_mask 已忽略")

        opts = NeoOptions(
            eta_ancestral=eta, eta_ddim=eta_ddim, s_churn=s_churn, s_tmin=s_tmin, s_tmax=s_tmax, s_noise=s_noise,
            sigma_min=sigma_min, sigma_max=sigma_max, rho=rho, always_discard_next_to_last_sigma=always_discard_penultimate_sigma,
            sgm_noise_multiplier=sgm_noise_multiplier, beta_dist_alpha=beta_alpha, beta_dist_beta=beta_beta,
            skip_early_cond=skip_early_cfg, s_min_uncond=ngms, s_min_uncond_all=ngms_all_steps, img2img_extra_noise=extra_noise,
        )

        samples = latent_image["samples"]
        samples = comfy.sample.fix_empty_latent_channels(model, samples)

        if shift > 0:
            model = _apply_shift(model, shift)

        is_txt2img = meta.get("mode") == "txt2img" and torch.count_nonzero(samples) == 0
        if turbo is not None:
            sampler_name, scheduler, cfg = turbo["sampler_name"], turbo["scheduler"], turbo["cfg"]
            steps = turbo["steps"] if is_txt2img else turbo["hires_steps"]
            logger.info(f"[Neo] Turbo: {sampler_name} / {scheduler} / {steps} 步 / CFG {cfg}")
        config = SAMPLERS[sampler_name]
        if is_txt2img and denoise < 1.0:
            logger.warning("[Neo] 文生图没有重绘幅度，denoise 已忽略")

        # ---- prompt schedules (processing.setup_conds) ----
        total_steps = config.total_steps(steps)
        base_steps, hires_steps = total_steps, None
        hires_mode = denoise_mode.startswith("Hires")
        if not is_txt2img and hires_mode and latent_image.get("neo_firstpass_steps"):
            base_steps, hires_steps = latent_image["neo_firstpass_steps"], total_steps

        clip_map = {}
        if prompt_lora:
            model, clip_map = _apply_prompt_loras(model, positive, negative)

        builder = _CondBuilder(clip_map)
        cond_parts = builder.expand(positive, True, base_steps, hires_steps)
        uncond_parts = None
        if cfg != 1:
            uncond_parts = builder.expand(negative, False, base_steps, hires_steps)
        else:
            logger.info("[Neo] CFG = 1.0 时忽略负面提示词（与 Neo 相同）")
        if not cond_parts:
            raise ValueError("Neo K采样器：正面条件为空")

        # ---- noise (processing.py: x = self.rng.next()) ----
        shape = tuple(samples.shape[1:])
        batch = samples.shape[0]
        if len(meta["seeds"]) != batch:
            meta = dict(meta)
            meta["seeds"] = [meta["seeds"][0] + (x if meta["subseed_strength"] == 0 else 0) for x in range(batch)]
            meta["subseeds"] = [meta["subseeds"][0] + x for x in range(batch)]
        rng = _rng_from_meta(meta, shape)
        if meta.get("state") is not None and tuple(meta.get("shape", ())) == shape and meta.get("batch") == batch:
            rng.set_state(meta["state"])  # continue the stream generated in Neo 空Latent
            noise = meta["noise"].to(rng.cfg.device)
        else:
            noise = rng.next()  # new shape (e.g. hires) -> new ImageRNG with the same seeds, like Neo

        # ---- model info ----
        base_model = model.model
        is_sdxl = isinstance(base_model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner))
        is_legacy = is_sdxl or type(base_model) is comfy.model_base.BaseModel
        predictor = PredictorAdapter(model.get_model_object("model_sampling"))

        run = NeoRun(
            config=config, scheduler=scheduler, steps=steps, cfg_scale=cfg, opts=opts, rng=rng, seeds=meta["seeds"],
            predictor=predictor, is_sdxl=is_sdxl, is_legacy=is_legacy,
            width=samples.shape[-1] * 8, height=samples.shape[-2] * 8,
            cond_parts=cond_parts, uncond_parts=uncond_parts,
            img2img=not is_txt2img, denoising_strength=denoise, hires_mode=hires_mode,
        )

        if config.kind == "timesteps":
            sigmas_for_comfy = torch.tensor([1.0, 0.0])
        else:
            sigmas_for_comfy = run.get_sigmas(ScheduleLinker(predictor, is_sdxl), run.setup_img2img_steps(steps)[0] if run.img2img else steps)

        guider = comfy.samplers.CFGGuider(model)
        guider.inner_set_conds(builder.conds)
        guider.set_cfg(cfg)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        out = guider.sample(noise, samples, _Bridge(run), sigmas_for_comfy, denoise_mask=None, callback=callback, disable_pbar=disable_pbar, seed=meta["seeds"][0])
        out = out.to(comfy.model_management.intermediate_device())

        result = latent_image.copy()
        result.pop("noise_mask", None)
        result["samples"] = out
        new_meta = {k: v for k, v in meta.items() if k not in ("state", "noise", "shape", "batch")}
        new_meta["mode"] = "img2img"
        result["neo_rng"] = new_meta
        if is_txt2img:
            result["neo_firstpass_steps"] = total_steps
        return (result,)


NODE_CLASS_MAPPINGS = {
    "NeoKSampler": NeoKSampler,
    "NeoCLIPConverter": NeoCLIPConverter,
    "NeoEmptyLatent": NeoEmptyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeoKSampler": "Neo K采样器 (Forge Neo)",
    "NeoCLIPConverter": "Neo CLIP 转换器 (Forge Neo)",
    "NeoEmptyLatent": "Neo 空Latent (Forge Neo 噪声)",
}
