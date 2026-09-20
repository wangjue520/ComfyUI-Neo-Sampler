"""ComfyUI-Neo-Sampler: Forge Neo (neo branch) sampling / noise / prompt-encoding replicated inside ComfyUI."""
import importlib
import logging
import subprocess
import sys

logger = logging.getLogger("NeoSampler")

# (pip name, import name) - installed automatically into the Python that runs ComfyUI
_REQUIREMENTS = [("lark", "lark"), ("torchsde", "torchsde"), ("scipy", "scipy")]


def _ensure_requirements():
    missing = []
    for pip_name, import_name in _REQUIREMENTS:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    if not missing:
        return
    logger.warning(f"[Neo] 正在自动安装依赖: {' '.join(missing)}")
    cmd = [sys.executable, "-s", "-m", "pip", "install", *missing]
    try:
        subprocess.check_call(cmd)
    except Exception as e:
        logger.error(f"[Neo] 自动安装失败，请手动执行: {' '.join(cmd)}\n{e}")
        raise
    importlib.invalidate_caches()


_ensure_requirements()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
