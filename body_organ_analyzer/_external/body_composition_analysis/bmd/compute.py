from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import skimage.draw
import skimage.measure
import skimage.morphology
import skimage.transform
from body_composition_analysis import debug_mode_enabled, get_debug_dir
from body_composition_analysis.bmd.definition import (
    BMD,
    AlgorithmInfo,
    Point,
    RayCastSearchResults,
    RoiMeasurement,
    RoiSelectionResults,
)
from body_composition_analysis.body_regions.definition import BodyRegion
from scipy.ndimage import gaussian_filter


def draw_line(
    bone_mask: np.ndarray, sample_pos1: np.ndarray, sample_pos2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_value = np.array([bone_mask.shape[0] - 1, bone_mask.shape[1] - 1])
    mask = np.zeros(bone_mask.shape, dtype=bool)
    rr, cc = skimage.draw.line(
        *np.minimum(sample_pos1.round().astype(int), max_value),
        *np.minimum(sample_pos2.round().astype(int), max_value),
    )
    mask[rr, cc] = True

    return np.logical_and(mask, bone_mask), rr, cc


def search_optimal_ray_angle(
    bone_mask: np.ndarray,
    image_data: np.ndarray,
    spinal_cord_center: np.ndarray,
    filter_sigma: float = 7.0,
    ray_length_mm: float = 100,
    debug: Optional[RayCastSearchResults] = None,
) -> float:
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    counts = []

    if debug is not None:
        debug.start_point = Point(
            int(round(spinal_cord_center[1])), int(round(spinal_cord_center[0]))
        )
        debug.end_points = []

    for phi in angles:
        sample_dir = np.array([np.sin(phi), np.cos(phi)])
        sample_pos1 = spinal_cord_center
        sample_pos2 = spinal_cord_center + ray_length_mm * sample_dir

        mask, rr, cc = draw_line(bone_mask, sample_pos1, sample_pos2)
        props = skimage.measure.regionprops(
            skimage.measure.label(mask[rr, cc][:, np.newaxis])
        )
        if len(props) == 0:
            counts.append(0)
        else:
            if debug is not None:
                debug.end_points.append(
                    Point(
                        x=int(cc[props[0].bbox[2] - 1]),
                        y=int(rr[props[0].bbox[2] - 1]),
                    ),  # TODO Recalculate point based on percentage of intersected ray
                )
            px = image_data[
                rr[props[0].bbox[0] : props[0].bbox[2]],
                cc[props[0].bbox[0] : props[0].bbox[2]],
            ]
            counts.append(int((px < 400).sum()))

    # Get angle from gaussian smoothed curve. Multiple angles can contain the same
    # number, so the middle position has to be calculated.
    smoothed_counts = gaussian_filter(
        np.asarray(counts).astype(float), sigma=filter_sigma, mode="wrap"
    )
    max_count = smoothed_counts.max()
    idx_max_angle = np.argwhere(smoothed_counts == max_count).astype(int)[0]
    sample_angle: float = angles[idx_max_angle].mean()

    if debug is not None:
        debug.voxel_counts = counts
        debug.smoothed_voxel_counts = smoothed_counts
        debug.selected_angle = sample_angle
        debug.selected_indices = idx_max_angle.tolist()

    return sample_angle


def find_intersection_start_end_points_on_ray(
    mask: np.ndarray,
    ray_angle: float,
    ray_origin: np.ndarray,
    spacing: np.ndarray,
    ray_length_mm: float = 100,
    min_intersection_length: float = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    ray_dir = np.array([np.sin(ray_angle), np.cos(ray_angle)])
    ray_dir /= spacing
    sample_pos1 = ray_origin
    sample_pos2 = ray_origin + ray_length_mm * ray_dir

    intersection_mask, rr, cc = draw_line(mask, sample_pos1, sample_pos2)
    intersection_on_line = intersection_mask[rr, cc]
    labeled_intersection_line_mask = skimage.measure.label(
        intersection_on_line[:, np.newaxis]
    )
    props = skimage.measure.regionprops(labeled_intersection_line_mask)
    while len(props) > 1:
        if props[0].area < min_intersection_length:
            del props[0]
        else:
            break
    pos = np.argwhere(
        labeled_intersection_line_mask == props[0].label
    )  # TODO Connected component analysis?

    start_pos = np.asarray([rr[pos[0, 0]], cc[pos[0, 0]]], dtype=float)
    end_pos = np.asarray([rr[pos[-1, 0]], cc[pos[-1, 0]]], dtype=float)

    return start_pos, end_pos


def find_roi_center(start_pos: np.ndarray, end_pos: np.ndarray) -> np.ndarray:
    return start_pos + 0.5 * (end_pos - start_pos)


def create_roi_mask(
    size: Sequence[int], center: np.ndarray, sample_angle: float, spacing: np.ndarray
) -> np.ndarray:
    coords = np.stack(
        np.meshgrid(
            *[np.arange(s) for s in size],
            indexing="ij",
        ),
        -1,
    )
    oval_adjustment = np.array([[1.5, 1.0]])
    distances = np.sqrt(
        np.square(oval_adjustment * spacing * (coords - center)).sum(-1)
    )
    distances = skimage.transform.rotate(
        distances, np.rad2deg(-sample_angle) + 90, center=center[::-1], mode="edge"
    )
    return distances < 10


def create_info_for_roi(image: np.ndarray, roi: np.ndarray) -> RoiMeasurement:
    masked_image = image[roi]
    num_dense_voxels = (masked_image > 500).sum()

    return RoiMeasurement(
        density_mean=float(masked_image.mean()),
        density_std=float(masked_image.std()),
        density_median=float(np.median(masked_image)),
        density_min=int(masked_image.min()),
        density_max=int(masked_image.max()),
        num_dense_roi_voxels=int(num_dense_voxels),
        num_roi_voxels=int(roi.sum()),
    )


def process_sample(
    image: sitk.Image,
    region_seg: sitk.Image,
    vertebrae_slice_mapping: Mapping[str, Sequence[int]],
) -> BMD:
    sx, sy, _ = image.GetSpacing()
    image_data = sitk.GetArrayViewFromImage(image)
    region_data = sitk.GetArrayViewFromImage(region_seg)

    result = BMD()

    for vid, (min_z, max_z) in vertebrae_slice_mapping.items():
        slice_idx = (min_z + max_z) // 2

        spinal_cord_mask = np.equal(
            region_data[slice_idx], BodyRegion.NERVOUS_SYSTEM.value
        )

        if spinal_cord_mask.sum() == 0:
            result.errors[vid] = "error_spinal_cord_not_found"
            continue

        # TODO: Add logic for T vertebrae if needed
        # TODO: Is there a way to make this work for C vertebrae?
        if vid.lower() == "l5":
            sccy, sccx = skimage.measure.regionprops(spinal_cord_mask.astype(np.uint8))[
                0
            ].centroid
        else:
            _, sccy, sccx = skimage.measure.regionprops(
                (
                    region_data[min_z : max_z + 1] == BodyRegion.NERVOUS_SYSTEM.value
                ).astype(np.uint8)
            )[0].centroid

        dilated_spinal_cord_mask = skimage.morphology.binary_dilation(spinal_cord_mask)

        bone_mask = np.equal(region_data[slice_idx], BodyRegion.BONE.value)
        bone_mask_labels = skimage.measure.label(bone_mask)
        bone_mask = np.isin(
            bone_mask_labels, np.unique(bone_mask_labels[dilated_spinal_cord_mask])[1:]
        )
        if len(image_data[slice_idx][bone_mask]) == 0:
            result.errors[vid] = "error_no_bone_found"
            continue

        if image_data[slice_idx][bone_mask].max() > 2500:
            result.errors[vid] = "error_metal_detected"
            continue

        # Second landmark: Ray-cast bone structure
        algorithm_results = AlgorithmInfo(
            selected_slice_idx=slice_idx,
            min_slice_idx=min_z,
            max_slice_idx=max_z,
            ray_cast_search=RayCastSearchResults(),
            roi_selection_results=RoiSelectionResults(),
        )
        sample_angle = search_optimal_ray_angle(
            np.logical_or(bone_mask, spinal_cord_mask),
            image_data[slice_idx],
            np.asarray([sccy, sccx]),
            debug=algorithm_results.ray_cast_search,
        )
        pos_bone_min, pos_bone_max = find_intersection_start_end_points_on_ray(
            mask=bone_mask,
            ray_angle=sample_angle,
            spacing=np.array([sy, sx]),
            ray_origin=np.asarray([sccy, sccx]),
        )
        roi_center = find_roi_center(pos_bone_min, pos_bone_max)
        algorithm_results.roi_selection_results.roi_center = Point(
            x=int(round(roi_center[1])), y=int(round(roi_center[0]))
        )
        algorithm_results.roi_selection_results.first_bone_pos = Point(
            x=int(pos_bone_min[1]), y=int(pos_bone_min[0])
        )
        algorithm_results.roi_selection_results.last_bone_pos = Point(
            x=int(pos_bone_max[1]), y=int(pos_bone_max[0])
        )
        result.algorithm[vid] = algorithm_results
        roi_mask = create_roi_mask(
            bone_mask.shape, roi_center, sample_angle, np.array([sy, sx])
        )

        result.measurements[vid] = create_info_for_roi(image_data[slice_idx], roi_mask)

        if debug_mode_enabled():
            from body_composition_analysis.report.plots.bmd import (
                create_debug_plot_for_vertebra,
            )

            create_debug_plot_for_vertebra(
                image, region_seg, vid, result.measurements[vid], result.algorithm[vid]
            ).savefig(get_debug_dir() / f"bmd_{vid.lower()}.png", bbox_inches="tight")

    return result
