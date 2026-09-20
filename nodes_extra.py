import logging
import math
from collections import namedtuple

import numpy as np
import torch
from PIL import Image

import comfy.model_management
import comfy.sd
import comfy.utils
import folder_paths

from .neo_core.sampler import SAMPLER_NAMES, SCHEDULER_NAMES
from .nodes import CATEGORY, _find_lora, _load_lora_sd, _parse_prompt_loras

logger = logging.getLogger("NeoSampler")

OOM_ERRORS = (getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError), torch.cuda.OutOfMemoryError)

LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

# modules/shared.py latent_upscale_modes
LATENT_UPSCALE_MODES = {
    "Latent": {"mode": "bilinear", "antialias": False},
    "Latent (antialiased)": {"mode": "bilinear", "antialias": True},
    "Latent (bicubic)": {"mode": "bicubic", "antialias": False},
    "Latent (bicubic antialiased)": {"mode": "bicubic", "antialias": True},
    "Latent (nearest)": {"mode": "nearest", "antialias": False},
    "Latent (nearest-exact)": {"mode": "nearest-exact", "antialias": False},
}
UPSCALER_MODEL = "放大模型（接 upscale_model）"
UPSCALER_NONE = "None（Lanczos）"


# region Neo upscaler (modules/images.py, modules/upscaler.py, modules/upscaler_utils.py)

Grid = namedtuple("Grid", ["tiles", "tile_w", "tile_h", "image_w", "image_h", "overlap"])


def split_grid(image, tile_w=512, tile_h=512, overlap=64):
    w, h = image.size

    non_overlap_width = tile_w - overlap
    non_overlap_height = tile_h - overlap

    cols = math.ceil((w - overlap) / non_overlap_width)
    rows = math.ceil((h - overlap) / non_overlap_height)

    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    grid = Grid([], tile_w, tile_h, w, h, overlap)
    for row in range(rows):
        row_images = []

        y = int(row * dy)

        if y + tile_h >= h:
            y = h - tile_h

        for col in range(cols):
            x = int(col * dx)

            if x + tile_w >= w:
                x = w - tile_w

            tile = image.crop((x, y, x + tile_w, y + tile_h))

            row_images.append([x, tile_w, tile])

        grid.tiles.append([y, tile_h, row_images])

    return grid


def combine_grid(grid):
    def make_mask_image(r):
        r = r * 255 / grid.overlap
        r = r.astype(np.uint8)
        return Image.fromarray(r, "L")

    mask_w = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((1, grid.overlap)).repeat(grid.tile_h, axis=0))
    mask_h = make_mask_image(np.arange(grid.overlap, dtype=np.float32).reshape((grid.overlap, 1)).repeat(grid.image_w, axis=1))

    combined_image = Image.new("RGB", (grid.image_w, grid.image_h))
    for y, h, row in grid.tiles:
        combined_row = Image.new("RGB", (grid.image_w, h))
        for x, w, tile in row:
            if x == 0:
                combined_row.paste(tile, (0, 0))
                continue

            combined_row.paste(tile.crop((0, 0, grid.overlap, h)), (x, 0), mask=mask_w)
            combined_row.paste(tile.crop((grid.overlap, 0, w, h)), (x + grid.overlap, 0))

        if y == 0:
            combined_image.paste(combined_row, (0, 0))
            continue

        combined_image.paste(combined_row.crop((0, 0, combined_row.width, grid.overlap)), (0, y), mask=mask_h)
        combined_image.paste(combined_row.crop((0, grid.overlap, combined_row.width, h)), (0, y + grid.overlap))

    return combined_image


def pil_image_to_torch_bgr(img):
    img = np.array(img.convert("RGB"))
    img = img[:, :, ::-1]
    img = np.transpose(img, (2, 0, 1))
    img = np.ascontiguousarray(img) / 255
    return torch.from_numpy(img)


def torch_bgr_to_pil_image(tensor):
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().float().cpu().numpy()
    arr = 255.0 * np.moveaxis(arr, 0, 2)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = arr[:, :, ::-1]
    return Image.fromarray(arr, "RGB")


def _param(model):
    return next(iter(model.model.parameters()))


def upscale_pil_patch(model, img):
    param = _param(model)
    with torch.inference_mode():
        tensor = pil_image_to_torch_bgr(img).unsqueeze(0)
        tensor = tensor.to(device=param.device, dtype=param.dtype)
        return torch_bgr_to_pil_image(model(tensor))


def upscale_with_model_cpu(model, img, tile_size, tile_overlap):
    if tile_size <= 0:
        return upscale_pil_patch(model, img)

    grid = split_grid(img, tile_size, tile_size, tile_overlap)
    newtiles = []
    scale_factor = 1
    pbar = comfy.utils.ProgressBar(sum(len(r[2]) for r in grid.tiles))
    for y, h, row in grid.tiles:
        newrow = []
        for x, w, tile in row:
            comfy.model_management.throw_exception_if_processing_interrupted()
            output = upscale_pil_patch(model, tile)
            scale_factor = output.width // tile.width
            newrow.append([x * scale_factor, w * scale_factor, output])
            pbar.update(1)
        newtiles.append([y * scale_factor, h * scale_factor, newrow])

    newgrid = Grid(newtiles, tile_w=grid.tile_w * scale_factor, tile_h=grid.tile_h * scale_factor, image_w=grid.image_w * scale_factor, image_h=grid.image_h * scale_factor, overlap=grid.overlap * scale_factor)
    return combine_grid(newgrid)


def do_upscale(model, img, tile_size, tile_overlap):
    """Neo's esrgan do_upscale; if VRAM runs out the tile size is halved automatically (Neo would just fail)."""
    size = tile_size
    while True:
        try:
            return upscale_with_model_cpu(model, img, size, tile_overlap)
        except OOM_ERRORS:
            comfy.model_management.soft_empty_cache()
            if size <= 64:
                raise
            size = max(64, size // 2)
            logger.warning(f"[Neo] 放大模型显存不足，分块缩小到 {size}")


def upscaler_upscale(model, img, scale, tile_size, tile_overlap):
    """modules/upscaler.py Upscaler.upscale"""
    dest_w = round(img.width * scale / 8) * 8
    dest_h = round(img.height * scale / 8) * 8

    for _ in range(4):
        _orig = img.size
        img = do_upscale(model, img, tile_size, tile_overlap)
        if (img.width >= dest_w and img.height >= dest_h) or img.size == _orig:
            break

    if (img.width != dest_w) or (img.height != dest_h):
        img = img.resize((dest_w, dest_h), LANCZOS)

    return img


def resize_image(im, width, height, model, tile_size, tile_overlap):
    """modules/images.py resize_image(resize_mode=0, ...)"""
    if model is None:
        return im.resize((width, height), resample=LANCZOS)

    scale = max(width / im.width, height / im.height)
    if scale > 1.0:
        im = upscaler_upscale(model, im, scale, tile_size, tile_overlap)

    if im.width != width or im.height != height:
        im = im.resize((width, height), resample=LANCZOS)

    return im


def s_round(val, step):
    """modules/ui.py sRound"""
    return math.floor(val / step + 0.5) * step


# endregion


class NeoPromptLoRA:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "clip": ("CLIP",), "text": ("STRING", {"forceInput": True, "tooltip": "正面提示词（与送进文本编码节点的是同一段文字）"})}}

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "按 Neo 规则解析提示词里的 <lora:名称:权重> 并加载到模型和 CLIP，结果与 LoRA 加载器逐比特一致。用了这个节点，Neo K采样器的 prompt_lora 请关闭。"

    def apply(self, model, clip, text):
        for name, te, unet in _parse_prompt_loras(text):
            rel = _find_lora(name)
            if rel is None:
                logger.error(f"[Neo] 找不到 LoRA「{name}」（与 Neo 相同：跳过）")
                continue
            logger.info(f"[Neo] 提示词 LoRA: {rel}  unet={unet} te={te}")
            model, clip = comfy.sd.load_lora_for_models(model, clip, _load_lora_sd(rel), unet, te)
        return (model, clip)


class NeoTurbo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "Anima Turbo LoRA（circlestone-labs 官方，Civitai 2560840）"}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "官方说明：略低于 1 会增加变化"}),
                "sampler_name": (SAMPLER_NAMES, {"default": "Euler"}),
                "scheduler": (SCHEDULER_NAMES, {"default": "Simple"}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "文生图步数（官方建议 8~12）"}),
                "hires_steps": ("INT", {"default": 8, "min": 1, "max": 100, "tooltip": "高清修复/图生图步数"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "官方建议 1（此时负面提示词不起作用）"}),
            }
        }

    RETURN_TYPES = ("MODEL", "NEO_TURBO")
    RETURN_NAMES = ("MODEL", "turbo")
    FUNCTION = "apply"
    CATEGORY = CATEGORY
    DESCRIPTION = "加载 Turbo LoRA（仅模型）并把 Neo K采样器切换到少步数、低 CFG。整组旁路时模型原样通过，采样器恢复自己的设置。"

    def apply(self, model, lora_name, strength, sampler_name, scheduler, steps, hires_steps, cfg):
        model, _ = comfy.sd.load_lora_for_models(model, None, _load_lora_sd(lora_name), strength, 0)
        return (model, {"sampler_name": sampler_name, "scheduler": scheduler, "steps": steps, "hires_steps": hires_steps, "cfg": cfg})


class NeoHiresUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "upscaler": (list(LATENT_UPSCALE_MODES) + [UPSCALER_MODEL, UPSCALER_NONE], {"default": UPSCALER_MODEL}),
                "scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 8.0, "step": 0.05, "tooltip": "Hires upscale by（Anima 单次建议 ≤1.5）"}),
                "resize_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Resize width to（0 = 按倍数）"}),
                "resize_height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Resize height to（0 = 按倍数）"}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
                "res_step": ("INT", {"default": 64, "min": 8, "max": 256, "step": 8, "tooltip": "Neo 设置 → Resolution Step（默认 64）"}),
                "tile_size": ("INT", {"default": 256, "min": 0, "max": 2048, "step": 16, "tooltip": "Neo 设置 → Tile Size for Upscalers；显存不足时会自动减半"}),
                "tile_overlap": ("INT", {"default": 16, "min": 0, "max": 256, "step": 4}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = CATEGORY
    DESCRIPTION = "复刻 Neo 高清修复的放大步骤（目标尺寸取整、Latent 插值或 放大模型→Lanczos→VAE 编码），输出直接接第二个 Neo K采样器（Hires.fix 模式）。VAE 与放大模型在显存不足时自动分块。"

    def upscale(self, latent, vae, upscaler, scale, resize_width, resize_height, upscale_model=None, res_step=64, tile_size=256, tile_overlap=16):
        samples = latent["samples"]
        width, height = samples.shape[-1] * 8, samples.shape[-2] * 8

        # StableDiffusionProcessingTxt2Img.calculate_target_resolution
        if resize_width == 0 and resize_height == 0:
            tw, th = s_round(width * scale, res_step), s_round(height * scale, res_step)
        else:
            if resize_height == 0:
                tw, th = resize_width, resize_width * (height / width)
            elif resize_width == 0:
                tw, th = resize_height * (width / height), resize_height
            else:
                tw, th = resize_width, resize_height
            tw, th = s_round(tw, res_step), s_round(th, res_step)
        tw, th = int(tw), int(th)
        logger.info(f"[Neo] 高清修复：{width}x{height} → {tw}x{th}（{upscaler}）")

        if upscaler in LATENT_UPSCALE_MODES:
            m = LATENT_UPSCALE_MODES[upscaler]
            s = samples
            _5d = s.ndim == 5
            if _5d:
                s = s.squeeze(2)
            s = torch.nn.functional.interpolate(s, size=(th // 8, tw // 8), mode=m["mode"], antialias=m["antialias"])
            if _5d:
                s = s.unsqueeze(2)
        else:
            model = None
            if upscaler == UPSCALER_MODEL:
                if upscale_model is None:
                    raise ValueError("Neo 高清修复：选了放大模型但没有接 upscale_model")
                model = upscale_model
                device = comfy.model_management.get_torch_device()
                comfy.model_management.free_memory(comfy.model_management.module_size(model.model) * 3 + 512 * 1024 * 1024, device)
                model.to(device)

            images = vae.decode(samples)
            if images.ndim == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])

            out = []
            try:
                for img in images:
                    arr = (255.0 * img.cpu().numpy()).astype(np.uint8)  # processing.py (truncating cast)
                    pil = resize_image(Image.fromarray(arr), tw, th, model, tile_size, tile_overlap)
                    out.append(np.array(pil).astype(np.float32) / 255.0)
            finally:
                if model is not None:
                    model.to("cpu")

            pixels = torch.from_numpy(np.stack(out))
            s = vae.encode(pixels[:, :, :, :3])

        result = latent.copy()
        result.pop("noise_mask", None)
        result["samples"] = s
        return (result,)


NODE_CLASS_MAPPINGS = {
    "NeoPromptLoRA": NeoPromptLoRA,
    "NeoTurbo": NeoTurbo,
    "NeoHiresUpscale": NeoHiresUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeoPromptLoRA": "Neo 提示词 LoRA (Forge Neo)",
    "NeoTurbo": "Neo Turbo 加速 (Forge Neo)",
    "NeoHiresUpscale": "Neo 高清修复放大 (Forge Neo)",
}
