import matplotlib.collections as mc
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import skimage.draw
import skimage.morphology
from body_composition_analysis.bmd.compute import create_roi_mask
from body_composition_analysis.bmd.definition import (
    AlgorithmInfo,
    Point,
    RayCastSearchResults,
    RoiMeasurement,
)
from body_composition_analysis.body_regions.definition import BodyRegion
from matplotlib.axes import Axes


def _crop_image(
    image: np.ndarray,
    ref_image: sitk.Image,
    center: Point,
    width: int = 100,
    height: int = 100,
) -> np.ndarray:
    sx, sy, _ = ref_image.GetSpacing()
    h = int(round(height / sy))
    w = int(round(width / sx))
    y1 = int(round(center.y - h * 0.5))
    x1 = int(round(center.x - w * 0.5))
    return image[y1 : y1 + h, x1 : x1 + w]


def create_roi_overlay(
    image: sitk.Image, algorithm: AlgorithmInfo, crop: bool = True
) -> np.ndarray:
    sx, sy, _ = image.GetSpacing()
    image_data = sitk.GetArrayViewFromImage(image)[
        algorithm.selected_slice_idx, ..., np.newaxis
    ]
    image_data = np.clip((image_data + 150) / 650, 0, 1)
    roi_mask = create_roi_mask(
        (image.GetHeight(), image.GetWidth()),
        (
            algorithm.roi_selection_results.roi_center.y,
            algorithm.roi_selection_results.roi_center.x,
        ),
        algorithm.ray_cast_search.selected_angle,
        (sy, sx),
    )
    roi_contour = roi_mask ^ skimage.morphology.binary_dilation(
        roi_mask, skimage.morphology.disk(1)
    )

    overlay_color = np.asarray([252, 129, 129]) / 255.0  # FC8181
    overlayed_image = np.where(
        roi_mask[..., np.newaxis],
        0.25 * roi_mask[..., np.newaxis] * overlay_color + 0.75 * image_data,
        image_data,
    )
    overlay_color = np.asarray([45, 55, 72]) / 255.0  # 2D3748
    overlayed_image = np.where(
        roi_contour[..., np.newaxis],
        0.75 * roi_contour[..., np.newaxis] * overlay_color + 0.25 * overlayed_image,
        overlayed_image,
    )
    overlayed_image = (overlayed_image * 255).astype(np.uint8)

    if not crop:
        return overlayed_image
    return _crop_image(overlayed_image, image, algorithm.ray_cast_search.start_point)


def _plot_segmentation(
    ax: Axes,
    image: sitk.Image,
    body_regions: sitk.Image,
    info: AlgorithmInfo,
    alpha: float = 0.75,
) -> None:
    image_data = sitk.GetArrayViewFromImage(image)[
        info.selected_slice_idx, ..., np.newaxis
    ]
    image_data = np.clip((image_data + 150) / 650, 0, 1)
    seg_data = sitk.GetArrayViewFromImage(body_regions)[info.selected_slice_idx]
    bone_mask = seg_data == BodyRegion.BONE.value
    spinal_cord_mask = seg_data == BodyRegion.NERVOUS_SYSTEM.value
    new_seg_data = np.zeros_like(seg_data)
    new_seg_data[bone_mask] = 1
    new_seg_data[spinal_cord_mask] = 2

    lut = np.asarray([[0, 0, 0], [252, 129, 129], [45, 55, 72]]) / 255

    overlayed_image = np.where(
        new_seg_data[..., np.newaxis] > 0,
        alpha * lut[new_seg_data] + (1 - alpha) * image_data,
        image_data,
    )
    overlayed_image = (overlayed_image * 255).astype(np.uint8)
    ax.imshow(overlayed_image)


def _plot_overview(ax: Axes, image: sitk.Image, info: AlgorithmInfo) -> None:
    image_data = create_roi_overlay(image, info, crop=False)
    ax.imshow(image_data)
    ax.plot(
        [
            info.roi_selection_results.first_bone_pos.x,
            info.roi_selection_results.last_bone_pos.x,
        ],
        [
            info.roi_selection_results.first_bone_pos.y,
            info.roi_selection_results.last_bone_pos.y,
        ],
        "--",
        color="#2D3748",
    )
    ax.plot(
        [
            info.roi_selection_results.first_bone_pos.x,
            info.roi_selection_results.last_bone_pos.x,
        ],
        [
            info.roi_selection_results.first_bone_pos.y,
            info.roi_selection_results.last_bone_pos.y,
        ],
        ".",
        color="#FC8181",
    )
    ax.plot(
        [
            info.ray_cast_search.start_point.x,
        ],
        [info.ray_cast_search.start_point.y],
        ".",
        color="#FC8181",
    )


def _plot_ray_casting(
    ax: Axes, image: np.ndarray, results: RayCastSearchResults
) -> None:
    ax.imshow(image, cmap="gray", vmin=-150, vmax=500)
    ax.add_collection(
        mc.LineCollection(
            [
                [
                    (results.start_point.x, results.start_point.y),
                    (end_point.x, end_point.y),
                ]
                for end_point in results.end_points
            ],
            colors=[
                "#2D3748" if i in results.selected_indices else "#FC8181"
                for i in range(len(results.end_points))
            ],
            alpha=0.75,
            linewidth=0.5,
        )
    )
    ax.plot(
        [results.start_point.x],
        [results.start_point.y],
        ".",
        color="#2D3748",
    )


def _plot_angle_curves(ax: Axes, results: RayCastSearchResults) -> None:
    angles = np.linspace(0, 2 * np.pi, len(results.end_points), endpoint=False)
    ax.plot(angles, results.voxel_counts, "-", color="#FC8181")
    ax.plot(
        angles,
        results.smoothed_voxel_counts,
        "-",
        color="#2D3748",
        alpha=0.75,
    )
    ax.axvline(results.selected_angle, linestyle="--", color="#2D3748", alpha=0.75)
    ax.set_xticks([0, np.pi / 2, np.pi, np.pi * 1.5, np.pi * 2])
    ax.set_xticklabels(["0°", "90°", "180°", "270°", "360°"])


def create_debug_plot_for_vertebra(
    image: sitk.Image,
    body_regions: sitk.Image,
    name: str,
    measurement: RoiMeasurement,
    info: AlgorithmInfo,
    crop_width: int = 100,
    crop_height: int = 100,
) -> plt.Figure:
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.5))
    _plot_segmentation(axs[0], image, body_regions, info)
    _plot_overview(axs[1], image, info)
    _plot_ray_casting(
        axs[2],
        sitk.GetArrayViewFromImage(image)[info.selected_slice_idx],
        info.ray_cast_search,
    )
    _plot_angle_curves(axs[3], info.ray_cast_search)

    axs[0].set_title("Body Region Segmentation", fontsize=10)
    axs[1].set_title(
        f"{name}: {measurement.density_mean:.2f}±{measurement.density_std:.2f} HU",
        fontsize=10,
    )
    axs[2].set_title(
        f"Slice {info.selected_slice_idx} ∈ [{info.min_slice_idx}, {info.max_slice_idx})",
        fontsize=10,
    )
    axs[3].set_title(
        f"Selected Angle: {np.rad2deg(info.ray_cast_search.selected_angle):.1f}°",
        fontsize=10,
    )

    sx, sy, _ = image.GetSpacing()
    h = int(round(crop_height / sy))
    w = int(round(crop_width / sx))
    y1 = int(round(info.ray_cast_search.start_point.y - h * 0.5))
    x1 = int(round(info.ray_cast_search.start_point.x - w * 0.5))
    for ax in axs[:-1]:
        ax.set_xlim(x1, x1 + w)
        ax.set_ylim(y1 + h, y1)

    return fig
