import logging
import os
from typing import Optional, Set

from body_organ_analysis.compute.constants import ALL_MODELS

logger = logging.getLogger(__name__)


def resolve_models(spec: Optional[str]) -> Set[str]:
    if not spec or spec.lower() == "all":
        return set(ALL_MODELS)
    models = set(spec.split("+"))
    invalid = models - ALL_MODELS
    if invalid:
        logger.error(
            f"Ignoring invalid model entries: {invalid}. "
            f"Available models are: {sorted(ALL_MODELS)}."
        )
        models -= invalid
    if "bca" in models:
        models.add("total")
    return models


def resolve_device(device: Optional[str] = None) -> str:
    device = device or os.environ.get("DEVICE", "gpu")
    device, _, gpu_id = device.partition(":")
    # TotalSegmentator's public API expects "gpu"; accept "cuda" as an alias.
    if device == "cuda":
        device = "gpu"
    gpu_id = gpu_id or os.environ.get("NVIDIA_ID", "")
    if gpu_id and device == "gpu":
        os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", gpu_id)
        device = f"gpu:{gpu_id}"
    return device
