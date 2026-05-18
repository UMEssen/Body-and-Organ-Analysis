import logging
import os

from body_organ_analysis.compute.constants import ALL_MODELS

logger = logging.getLogger(__name__)


def env_bool(name: str, default: bool = False) -> bool:
    """Parse an environment variable as a boolean."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true"}


def resolve_models(spec: str | None) -> set[str]:
    if not spec or spec.lower() == "all":
        return set(ALL_MODELS)
    models = {s.replace("-", "_") for s in spec.split("+")}
    invalid = models - ALL_MODELS
    if invalid:
        logger.error(
            "Ignoring invalid model entries: %s. Available models are: %s.",
            invalid,
            sorted(ALL_MODELS),
        )
        models -= invalid
    if "bca" in models:
        models.add("total")
    return models


def resolve_device(device: str | None = None) -> str:
    device_str = device or os.environ.get("DEVICE", "gpu")
    device_str, _, gpu_id = device_str.partition(":")
    # TotalSegmentator's public API expects "gpu"; accept "cuda" as an alias.
    if device_str == "cuda":
        device_str = "gpu"
    gpu_id = gpu_id or os.environ.get("NVIDIA_ID", "")
    if gpu_id and device_str == "gpu":
        os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", gpu_id)
        device_str = f"gpu:{gpu_id}"
    return device_str
