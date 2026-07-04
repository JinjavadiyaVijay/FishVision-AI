"""
device.py — GPU detection, memory monitoring, and device selection.

Provides a single source of truth for hardware capabilities and VRAM tracking
throughout the training pipeline.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Snapshot of GPU hardware and current memory state."""

    name: str
    total_memory_gb: float
    allocated_memory_gb: float
    reserved_memory_gb: float
    free_memory_gb: float
    utilization_pct: float
    cuda_version: str
    compute_capability: tuple[int, int]


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Select the best available device.

    Parameters
    ----------
    force_cpu : bool
        If True, always return CPU even if GPU is available.

    Returns
    -------
    torch.device
    """
    if force_cpu:
        logger.info("Device: CPU (forced)")
        return torch.device("cpu")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("Device: %s (%.1f GB VRAM)", name, mem_gb)
        return device

    logger.warning("CUDA not available — falling back to CPU (training will be very slow)")
    return torch.device("cpu")


def get_gpu_info() -> Optional[GPUInfo]:
    """
    Get detailed GPU information.

    Returns None if CUDA is not available.
    """
    if not torch.cuda.is_available():
        return None

    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9

    # Try to get utilization via pynvml (optional)
    utilization = 0.0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        utilization = float(util.gpu)
        pynvml.nvmlShutdown()
    except Exception:
        pass  # pynvml not available — utilization unknown

    return GPUInfo(
        name=torch.cuda.get_device_name(0),
        total_memory_gb=round(total, 2),
        allocated_memory_gb=round(allocated, 3),
        reserved_memory_gb=round(reserved, 3),
        free_memory_gb=round(total - reserved, 2),
        utilization_pct=utilization,
        cuda_version=torch.version.cuda or "unknown",
        compute_capability=(props.major, props.minor),
    )


def log_gpu_memory(tag: str = "") -> dict[str, float]:
    """
    Log current GPU memory usage and return as dict.

    Parameters
    ----------
    tag : str
        Optional label for the log message (e.g. "after model load").

    Returns
    -------
    dict with allocated_gb, reserved_gb, free_gb (empty dict if no GPU)
    """
    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved

    prefix = f"[{tag}] " if tag else ""
    logger.info(
        "%sGPU Memory: %.2f GB allocated, %.2f GB reserved, %.2f GB free / %.1f GB total",
        prefix, allocated, reserved, free, total,
    )

    return {
        "allocated_gb": round(allocated, 3),
        "reserved_gb": round(reserved, 3),
        "free_gb": round(free, 2),
    }


def get_system_info() -> dict[str, str]:
    """Collect system information for experiment logging."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = str(
            round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
        )
    return info
