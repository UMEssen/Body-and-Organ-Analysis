import base64

import cv2
import numpy as np

ADDITIONAL_MODELS_OUTPUT_NAME: dict[str, str] = {
    "lung_vessels": "lung_vessels_airways",
    "cerebral_bleed": "cerebral_bleed",
    "hip_implant": "hip_implant",
    "coronary_arteries": "coronary_arteries",
    "pleural_pericard_effusion": "pleural_pericard_effusion",
    "liver_vessels": "liver_vessels",
    "heartchambers_highres": "heartchambers",
}


def convert_resampling_slices(
    slices: int, current_sampling: float, target_resampling: float | None
) -> int:
    if target_resampling is None:
        return slices
    return round((slices / target_resampling) * current_sampling)


def create_mask(region_data: np.ndarray, labels: int | list[int]) -> np.ndarray:
    mask = np.zeros(region_data.shape, dtype=bool)
    if isinstance(labels, int):
        mask[region_data == labels] = True
    else:
        mask[np.isin(region_data, labels)] = True
    return mask


def convert_name(name: str) -> str:
    return "".join(s.capitalize() for s in name.split("_"))


def to_png_data_url(image: np.ndarray) -> str:
    """Encode an RGB ndarray as a base64 PNG data URL for HTML embedding."""
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    _, encoded = cv2.imencode(
        ".png", image[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 6]
    )
    return "data:image/png;base64," + base64.b64encode(encoded).decode("utf-8")
