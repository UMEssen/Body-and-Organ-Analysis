from typing import List, Tuple

import cv2
import numpy as np
import SimpleITK as sitk
import skimage.morphology
from body_composition_analysis.report.plots.colors import TISSUE_HEATMAP_COLOR_MAP
from body_composition_analysis.tissue.definition import Tissue


def _get_contour_image(
    body_regions: np.ndarray, axis: int, scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.logical_and(body_regions > 0, body_regions < 255)
    mask = mask.any(axis=axis)
    mask = cv2.resize(
        mask.astype(np.uint8),
        dsize=None,
        fx=1,
        fy=scale,
        interpolation=cv2.INTER_NEAREST,
    )
    mask = np.pad(mask, ((1, 1), (0, 0)), "constant")
    dilated_mask = skimage.morphology.binary_dilation(mask, skimage.morphology.disk(2))

    contour = dilated_mask & ~mask
    return contour[1:-1] > 0, mask[1:-1] > 0


def create_tissue_heatmaps(
    body_regions: sitk.Image, tissues: sitk.Image
) -> List[Tuple[Tissue, np.ndarray, np.ndarray]]:
    spacing = tissues.GetSpacing()
    region_data = sitk.GetArrayViewFromImage(body_regions)
    tissue_data = sitk.GetArrayViewFromImage(tissues)

    # TODO Filter tissues based on examined body region in CT scan
    selected_tissues = (
        Tissue.BONE,
        Tissue.MUSCLE,
        Tissue.IMAT,
        Tissue.SAT,
        Tissue.VAT,
        Tissue.PAT,
        Tissue.EAT,
    )

    # Precompute coronal and sagittal boolean body and contour masks
    coronal_contour, coronal_mask = _get_contour_image(
        region_data, 1, spacing[2] / spacing[1]
    )
    sagittal_contour, sagittal_mask = _get_contour_image(
        region_data, 2, spacing[2] / spacing[0]
    )

    # Create tissue heatmaps for each selected tissue
    result = []
    for tissue in selected_tissues:
        # Compute tissue density
        tissue_mask = tissue_data == tissue.value
        coronal_heatmap = tissue_mask.sum(axis=1)
        sagittal_heatmap = tissue_mask.sum(axis=2)

        # Normalize to maximum tissue density
        coronal_heatmap = coronal_heatmap / max(coronal_heatmap.max(), 1)
        sagittal_heatmap = sagittal_heatmap / max(sagittal_heatmap.max(), 1)

        # Compose coronal heatmap image
        coronal_heatmap = TISSUE_HEATMAP_COLOR_MAP(coronal_heatmap)[..., :3]
        coronal_heatmap = (coronal_heatmap * 255).astype(np.uint8)
        coronal_heatmap = cv2.resize(
            coronal_heatmap,
            dsize=None,
            fx=1,
            fy=spacing[2] / spacing[1],
            interpolation=cv2.INTER_NEAREST,
        )
        coronal_heatmap = np.where(~coronal_mask[..., np.newaxis], 255, coronal_heatmap)
        coronal_heatmap = np.where(coronal_contour[..., np.newaxis], 0, coronal_heatmap)
        coronal_heatmap = coronal_heatmap[::-1, ...]

        # Compose sagittal heatmap image
        sagittal_heatmap = TISSUE_HEATMAP_COLOR_MAP(sagittal_heatmap)[..., :3]
        sagittal_heatmap = (sagittal_heatmap * 255).astype(np.uint8)
        sagittal_heatmap = cv2.resize(
            sagittal_heatmap,
            dsize=None,
            fx=1,
            fy=spacing[2] / spacing[0],
            interpolation=cv2.INTER_NEAREST,
        )
        sagittal_heatmap = np.where(
            ~sagittal_mask[..., np.newaxis], 255, sagittal_heatmap
        )
        sagittal_heatmap = np.where(
            sagittal_contour[..., np.newaxis], 0, sagittal_heatmap
        )
        sagittal_heatmap = sagittal_heatmap[::-1, ...]

        result.append((tissue, coronal_heatmap, sagittal_heatmap))

    return result
