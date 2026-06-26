import cv2
import numpy as np


def blend_overlay(
    gray_image: np.ndarray,
    color_rgb: np.ndarray,
    mask: np.ndarray,
    opacity: float,
) -> np.ndarray:
    gray_rgb = gray_image[..., np.newaxis]
    blended = gray_rgb * (1 - opacity) + color_rgb.astype(np.float64) * opacity
    return np.where(mask[..., np.newaxis], blended, gray_rgb)


def resize_with_y_scale(image: np.ndarray, scale_y: float) -> np.ndarray:
    return cv2.resize(
        image, dsize=None, fx=1, fy=scale_y, interpolation=cv2.INTER_NEAREST
    )


def apply_hu_window(
    image: np.ndarray, hu_min: float = -150.0, hu_max: float = 400.0
) -> np.ndarray:
    return np.clip((image - hu_min) / (hu_max - hu_min), 0.0, 1.0) * 255.0
