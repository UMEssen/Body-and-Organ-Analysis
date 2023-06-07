from typing import Tuple

import cv2
import numpy as np
import SimpleITK as sitk
import skimage.morphology
from body_composition_analysis.report.plots.colors import TISSUE_COLOR_MAP


def create_aggregation_image(
    image: sitk.Image,
    body_regions: sitk.Image,
    tissues: sitk.Image,
    group: Tuple[int, int],
    opacity: float = 0.25,
) -> np.ndarray:
    # Access raw data
    spacing = image.GetSpacing()
    image_data = sitk.GetArrayViewFromImage(image)
    region_seg = sitk.GetArrayViewFromImage(body_regions)
    tissue_seg = sitk.GetArrayViewFromImage(tissues)

    # Compute body, selection and contour masks
    sag_index = region_seg.shape[2] // 2
    sag_body_mask = np.logical_and(
        region_seg[..., sag_index] > 0, region_seg[..., sag_index] < 255
    )
    sag_selection_mask = np.zeros_like(sag_body_mask)
    sag_selection_mask[group[0] : group[1]] = 1
    selection_contour = np.logical_xor(
        sag_selection_mask,
        skimage.morphology.binary_dilation(
            sag_selection_mask, skimage.morphology.disk(1)
        ),
    )

    # Apply hounsfield unit window and mask everything outside body
    sag_image = image_data[..., sag_index]
    sag_image = np.clip((sag_image + 150) / 400.0, 0.0, 1.0) * 255
    sag_image = np.where(sag_body_mask, sag_image, 255)

    # Apply tissue color map
    sag_tissue = tissue_seg[..., sag_index]
    sag_tissue = np.where(sag_selection_mask, sag_tissue, 0)
    sag_tissue_rgb = TISSUE_COLOR_MAP[sag_tissue]

    # Overlay colored tissues over the CT image
    composed = np.where(
        sag_body_mask[..., np.newaxis],
        sag_image[..., np.newaxis] * (1 - opacity) + sag_tissue_rgb * opacity,
        sag_image[..., np.newaxis],
    )

    # Mark outer selection contour outside of the body as black
    composed = np.where(
        np.logical_and(~sag_body_mask, selection_contour)[..., np.newaxis],
        np.array([[[0, 0, 0]]]),
        composed,
    )

    # Blend outer selection countour inside of the body with the composed image
    composed = np.where(
        np.logical_and(sag_body_mask, selection_contour)[..., np.newaxis],
        composed * (1 - opacity) + np.array([[[0, 0, 0]]]) * opacity,
        composed,
    )

    # Resize the composed image to the correct aspect ratio
    scale_factor = spacing[2] / spacing[1]
    composed = cv2.resize(
        composed, dsize=None, fx=1, fy=scale_factor, interpolation=cv2.INTER_NEAREST
    )

    # Flip image vertically
    return composed[::-1, :]
