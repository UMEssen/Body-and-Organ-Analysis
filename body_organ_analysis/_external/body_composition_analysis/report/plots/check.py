import numpy as np
import SimpleITK as sitk

from body_composition_analysis.report.plots.overlay import (
    apply_hu_window,
    blend_overlay,
)


def create_equidistant_overview(
    image: sitk.Image,
    segmentations: list[tuple[sitk.Image, dict[int, tuple[int, int, int]]]],
    opacity: float = 0.25,
) -> list[list[str | np.ndarray]]:
    image_arr = sitk.GetArrayViewFromImage(image)
    segmentations_arr = [(sitk.GetArrayViewFromImage(s), c) for s, c in segmentations]
    num_slices = image_arr.shape[0]
    locations = [
        0,
        int(num_slices * 0.25),
        int(num_slices * 0.5),
        int(num_slices * 0.75),
        num_slices - 1,
    ]
    names = ["First", "25%", "Central", "75%", "Last"]
    result = []
    for name, slice_idx in zip(names, locations, strict=False):
        slice_image = apply_hu_window(image_arr[slice_idx, ...])
        slices: list[str | np.ndarray] = [name]
        for seg, color_map in segmentations_arr:
            slice_seg = seg[slice_idx, ...]
            slice_seg_rgb = np.asarray(color_map)[slice_seg]
            composed = blend_overlay(slice_image, slice_seg_rgb, slice_seg > 0, opacity)
            slices.append(composed)
        result.append(slices)
    return result
