import logging
import os

from body_organ_analysis.compute.constants import (
    ALL_MODELS,
    AVAILABLE_MODELS,
    LICENSE_MODELS,
)

logger = logging.getLogger(__name__)


def env_bool(name: str, default: bool = False) -> bool:
    """Parse an environment variable as a boolean."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true"}


def env_str(name: str, default: str | None = None) -> str | None:
    """Parse an environment variable as a string."""
    raw = os.getenv(name)
    if raw is None or raw.strip().lower() in {"", "todo"}:
        return default
    return raw.strip()


def resolve_models(
    spec: str | None, strict: bool = False, license_number: str | None = None
) -> set[str]:
    if not spec or spec.lower() == "all":
        models = set(ALL_MODELS)
        if license_number:
            from totalsegmentator.config import is_valid_license  # noqa: PLC0415

            if is_valid_license(license_number):
                models |= LICENSE_MODELS
    else:
        models = {s.replace("-", "_") for s in spec.split("+")}
        invalid = models - AVAILABLE_MODELS
        if invalid:
            if strict:
                raise ValueError(
                    f"Unknown model(s): {', '.join(sorted(invalid))}. "
                    f"Available: {', '.join(sorted(AVAILABLE_MODELS))}"
                )
            logger.error(
                "Ignoring invalid model entries: %s. Available models are: %s.",
                invalid,
                sorted(AVAILABLE_MODELS),
            )
            models -= invalid
    if "bca" in models:
        models = (models | {"total"}) - {"body_regions", "body_parts"}
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
