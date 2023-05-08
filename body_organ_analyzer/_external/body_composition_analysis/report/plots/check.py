from typing import List, Tuple

import numpy as np
import SimpleITK as sitk
from body_composition_analysis.report.plots.colors import (
    BODY_REGION_COLOR_MAP,
    TISSUE_COLOR_MAP,
)


def create_equidistant_overview(
    image: sitk.Image, regions: sitk.Image, tissues: sitk.Image, opacity: float = 0.25
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    image = sitk.GetArrayViewFromImage(image)
    tissue_seg = sitk.GetArrayViewFromImage(tissues)
    region_seg = sitk.GetArrayViewFromImage(regions)

    num_slices = tissue_seg.shape[0]
    locations = [
        0,
        int(num_slices * 0.25),
        int(num_slices * 0.5),
        int(num_slices * 0.75),
        num_slices - 1,
    ]
    names = ["First", "25%", "Central", "75%", "Last"]

    result = []
    for name, slice_idx in zip(names, locations):
        slice_image = np.clip((image[slice_idx, ...] + 150) / 400.0, 0.0, 1.0) * 255
        slice_seg = tissue_seg[slice_idx, ...]
        slice_seg_rgb = TISSUE_COLOR_MAP[slice_seg]

        composed = np.where(
            slice_seg[..., np.newaxis] > 0,
            slice_image[..., np.newaxis] * (1 - opacity) + slice_seg_rgb * opacity,
            slice_image[..., np.newaxis],
        )

        slice_seg = region_seg[slice_idx, ...]
        slice_seg_rgb = BODY_REGION_COLOR_MAP[slice_seg]

        composed2 = np.where(
            slice_seg[..., np.newaxis] > 0,
            slice_image[..., np.newaxis] * (1 - opacity) + slice_seg_rgb * opacity,
            slice_image[..., np.newaxis],
        )

        result.append((name, composed, composed2))

    return result
