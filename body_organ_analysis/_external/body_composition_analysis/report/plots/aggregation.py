import numpy as np
import SimpleITK as sitk
from skimage.morphology import dilation, disk

from body_composition_analysis.report.plots.colors import TISSUE_COLOR_MAP
from body_composition_analysis.report.plots.overlay import (
    apply_hu_window,
    blend_overlay,
    resize_with_y_scale,
)


def create_aggregation_image(
    image: sitk.Image,
    body_regions: sitk.Image,
    tissues: sitk.Image,
    group: tuple[int, int],
    opacity: float = 0.25,
    theme: str = "light",
) -> np.ndarray:
    # Access raw data
    spacing = image.GetSpacing()
    image_data = sitk.GetArrayViewFromImage(image)
    region_seg = sitk.GetArrayViewFromImage(body_regions)
    tissue_seg = sitk.GetArrayViewFromImage(tissues)
    is_dark = theme == "dark"
    background = 0 if is_dark else 255
    contour = [255, 255, 255] if is_dark else [0, 0, 0]

    # Compute body, selection and contour masks
    sag_index = region_seg.shape[2] // 2
    sag_body_mask = np.logical_and(
        region_seg[..., sag_index] > 0, region_seg[..., sag_index] < 255
    )
    sag_selection_mask = np.zeros_like(sag_body_mask)
    sag_selection_mask[group[0] : group[1]] = 1
    selection_contour = np.logical_xor(
        sag_selection_mask,
        dilation(sag_selection_mask, disk(1)),
    )

    # Apply hounsfield unit window and mask everything outside body
    sag_image = apply_hu_window(image_data[..., sag_index])
    sag_image = np.where(sag_body_mask, sag_image, background)

    # Apply tissue color map
    sag_tissue = tissue_seg[..., sag_index]
    sag_tissue = np.where(sag_selection_mask, sag_tissue, 0)
    sag_tissue_rgb = TISSUE_COLOR_MAP[sag_tissue]

    # Overlay colored tissues over the CT image
    composed = blend_overlay(sag_image, sag_tissue_rgb, sag_body_mask, opacity)

    # Mark outer selection contour outside of the body as black
    composed = np.where(
        np.logical_and(~sag_body_mask, selection_contour)[..., np.newaxis],
        np.array([[contour]]),
        composed,
    )

    # Blend outer selection contour inside of the body with the composed image
    composed = np.where(
        np.logical_and(sag_body_mask, selection_contour)[..., np.newaxis],
        composed * (1 - opacity) + np.array([[[0, 0, 0]]]) * opacity,
        composed,
    )

    # Resize the composed image to the correct aspect ratio
    composed = resize_with_y_scale(composed, spacing[2] / spacing[1])

    # Flip image vertically
    return composed[::-1, :]
