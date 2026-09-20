# Port of Forge Neo modules/rng.py (neo branch).
# The algorithm is kept line-for-line; the only change is that Neo's global `shared.opts.randn_source`
# / `shared.opts.eta_noise_seed_delta` / `devices.device` are carried in an RNGConfig object instead.
import torch

from . import rng_philox


class RNGConfig:
    def __init__(self, source: str, device: torch.device, eta_noise_seed_delta: int = 0):
        assert source in ("CPU", "GPU", "NV")
        self.source = source
        self.device = torch.device(device)
        self.eta_noise_seed_delta = int(eta_noise_seed_delta or 0)
        self.nv_rng = None  # Neo keeps this as a module global

    @property
    def local_device(self):
        return torch.device("cpu") if self.source == "CPU" or self.device.type == "mps" else self.device


def randn(cfg: RNGConfig, seed, shape, generator=None):
    if generator is not None:
        manual_seed(cfg, (seed + 100000) % 65536)
    else:
        manual_seed(cfg, seed)

    if cfg.source == "NV":
        return torch.asarray((generator or cfg.nv_rng).randn(shape), device=cfg.device)

    if cfg.source == "CPU" or cfg.device.type == "mps":
        return torch.randn(shape, device=torch.device("cpu"), generator=generator).to(cfg.device)

    return torch.randn(shape, device=cfg.device, generator=generator)


def randn_local(cfg: RNGConfig, seed, shape):
    if cfg.source == "NV":
        rng = rng_philox.Generator(seed)
        return torch.asarray(rng.randn(shape), device=cfg.device)

    local_device = cfg.local_device
    local_generator = torch.Generator(local_device).manual_seed(int(seed))
    return torch.randn(shape, device=local_device, generator=local_generator).to(cfg.device)


def randn_without_seed(cfg: RNGConfig, shape, generator=None):
    if cfg.source == "NV":
        return torch.asarray((generator or cfg.nv_rng).randn(shape), device=cfg.device)

    if cfg.source == "CPU" or cfg.device.type == "mps":
        return torch.randn(shape, device=torch.device("cpu"), generator=generator).to(cfg.device)

    return torch.randn(shape, device=cfg.device, generator=generator)


def manual_seed(cfg: RNGConfig, seed):
    if cfg.source == "NV":
        cfg.nv_rng = rng_philox.Generator(seed)
        return

    torch.manual_seed(seed)


def create_generator(cfg: RNGConfig, seed):
    if cfg.source == "NV":
        return rng_philox.Generator(seed)

    generator = torch.Generator(cfg.local_device).manual_seed(int(seed))
    return generator


def slerp(val: float, low: torch.Tensor, high: torch.Tensor, eps=1e-6) -> torch.Tensor:
    b = low.shape[0]

    low_flat = low.reshape(b, -1)
    high_flat = high.reshape(b, -1)

    low_norm = low_flat / (low_flat.norm(dim=1, keepdim=True) + eps)
    high_norm = high_flat / (high_flat.norm(dim=1, keepdim=True) + eps)

    dot = (low_norm * high_norm).sum(dim=1).clamp(-1 + eps, 1 - eps)

    omega = torch.acos(dot)
    so = torch.sin(omega)

    mask = so.abs() < eps
    so = torch.where(mask, torch.ones_like(so), so)

    res_flat = torch.sin((1.0 - val) * omega).unsqueeze(1) / so.unsqueeze(1) * low_flat + torch.sin(val * omega).unsqueeze(1) / so.unsqueeze(1) * high_flat

    res_flat = torch.where(mask.unsqueeze(1), (1 - val) * low_flat + val * high_flat, res_flat)

    return res_flat.reshape_as(low)


class ImageRNG:
    def __init__(self, cfg: RNGConfig, shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0):
        self.cfg = cfg
        self.shape = tuple(map(int, shape))
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.seed_resize_from_h = seed_resize_from_h
        self.seed_resize_from_w = seed_resize_from_w

        self.generators = [create_generator(cfg, seed) for seed in seeds]

        self.is_first = True

    def first(self):
        cfg = self.cfg
        noise_shape = self.shape if self.seed_resize_from_h <= 0 or self.seed_resize_from_w <= 0 else (self.shape[0], int(self.seed_resize_from_h) // 8, int(self.seed_resize_from_w // 8))

        xs = []

        for i, (seed, generator) in enumerate(zip(self.seeds, self.generators)):
            subnoise = None
            if self.subseeds is not None and self.subseed_strength != 0:
                subseed = 0 if i >= len(self.subseeds) else self.subseeds[i]
                subnoise = randn(cfg, subseed, noise_shape)

            if noise_shape != self.shape:
                noise = randn(cfg, seed, noise_shape)
            else:
                noise = randn(cfg, seed, self.shape, generator=generator)

            if subnoise is not None:
                noise = slerp(self.subseed_strength, noise, subnoise)

            if noise_shape != self.shape:
                x = randn(cfg, seed, self.shape, generator=generator)
                dx = (self.shape[2] - noise_shape[2]) // 2
                dy = (self.shape[1] - noise_shape[1]) // 2
                w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
                h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
                tx = 0 if dx < 0 else dx
                ty = 0 if dy < 0 else dy
                dx = max(-dx, 0)
                dy = max(-dy, 0)

                x[:, ty : ty + h, tx : tx + w] = noise[:, dy : dy + h, dx : dx + w]
                noise = x

            xs.append(noise)

        eta_noise_seed_delta = cfg.eta_noise_seed_delta or 0
        if eta_noise_seed_delta:
            self.generators = [create_generator(cfg, seed + eta_noise_seed_delta) for seed in self.seeds]

        return torch.stack(xs).to(cfg.device)

    def next(self):
        if self.is_first:
            self.is_first = False
            return self.first()

        xs = []
        for generator in self.generators:
            x = randn_without_seed(self.cfg, self.shape, generator=generator)
            xs.append(x)

        return torch.stack(xs).to(self.cfg.device)

    # ---- additions (not in Neo): snapshot / restore so the noise can be generated in one node ----
    # ---- and the very same generator stream can be continued inside the sampler node.         ----
    def get_state(self):
        states = []
        for g in self.generators:
            if isinstance(g, rng_philox.Generator):
                states.append(("philox", int(g.seed), int(g.offset)))
            else:
                states.append(("torch", g.get_state().clone()))
        return {"is_first": self.is_first, "generators": states}

    def set_state(self, state):
        self.is_first = state["is_first"]
        gens = []
        for st in state["generators"]:
            if st[0] == "philox":
                g = rng_philox.Generator(st[1])
                g.offset = st[2]
            else:
                g = torch.Generator(self.cfg.local_device)
                g.set_state(st[1])
            gens.append(g)
        self.generators = gens
