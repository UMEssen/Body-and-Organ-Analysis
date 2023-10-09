from typing import List, Tuple, Dict, Union

import numpy as np
import SimpleITK as sitk


def create_equidistant_overview(
    image: sitk.Image,
    segmentations: List[Tuple[sitk.Image, Dict[int, Tuple[int, int, int]]]],
    opacity: float = 0.25,
) -> List[List[Union[str, np.ndarray]]]:
    image = sitk.GetArrayViewFromImage(image)
    segmentations_arr = [(sitk.GetArrayViewFromImage(s), c) for s, c in segmentations]
    num_slices = image.shape[0]
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

        slices = [name]
        for seg, color_map in segmentations_arr:
            slice_seg = seg[slice_idx, ...]
            slice_seg_rgb = color_map[slice_seg]

            composed = np.where(
                slice_seg[..., np.newaxis] > 0,
                slice_image[..., np.newaxis] * (1 - opacity) + slice_seg_rgb * opacity,
                slice_image[..., np.newaxis],
            )
            slices.append(composed)
        result.append(slices)
    return result
