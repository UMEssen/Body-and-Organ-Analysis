import numpy as np
import SimpleITK as sitk


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
        slice_image = np.clip((image_arr[slice_idx, ...] + 150) / 400.0, 0.0, 1.0) * 255
        slices: list[str | np.ndarray] = [name]
        for seg, color_map in segmentations_arr:
            slice_seg = seg[slice_idx, ...]
            slice_seg_rgb = np.asarray(color_map)[slice_seg]

            composed = np.where(
                slice_seg[..., np.newaxis] > 0,
                slice_image[..., np.newaxis] * (1 - opacity) + slice_seg_rgb * opacity,
                slice_image[..., np.newaxis],
            )
            slices.append(composed)
        result.append(slices)
    return result
