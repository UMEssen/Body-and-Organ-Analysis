import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from totalsegmentator.map_to_binary import reverse_class_map_complete

from body_organ_analysis.compute.util import ADDITIONAL_MODELS_OUTPUT_NAME, create_mask

logger = logging.getLogger(__name__)


def autochton_reference(
    ct_data: np.ndarray,
    autochthon_right_mask: np.ndarray,
    autochthon_left_mask: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    if np.sum(autochthon_right_mask) == 0 and np.sum(autochthon_left_mask) == 0:
        return None, None
    # TODO: Does it make sense to consider one if the other one is not there?
    # If only the right one is none
    if np.sum(autochthon_right_mask) == 0 and np.sum(autochthon_left_mask) > 0:
        return float(np.mean(ct_data[autochthon_left_mask])), float(
            np.std(ct_data[autochthon_left_mask])
        )
    # If only the left one is none
    if np.sum(autochthon_left_mask) == 0 and np.sum(autochthon_right_mask) > 0:
        return float(np.mean(ct_data[autochthon_right_mask])), float(
            np.std(ct_data[autochthon_right_mask])
        )

    autochton_mean = (
        np.mean(ct_data[autochthon_left_mask]) + np.mean(ct_data[autochthon_right_mask])
    ) / 2
    autochton_std = (
        np.std(ct_data[autochthon_left_mask]) + np.std(ct_data[autochthon_right_mask])
    ) / 2
    return float(autochton_mean), float(autochton_std)


def metrics_for_region(
    ct_data: np.ndarray,
    mask: np.ndarray,
    autochton_mean: Optional[float],
    autochton_std: Optional[float],
    img_spacing: np.ndarray,
) -> Dict[str, Any]:
    measurements: Dict[str, Any] = {}
    if np.sum(mask) == 0:
        measurements["present"] = False
        return measurements
    ml_per_voxel = np.prod(img_spacing) / 1000.0
    measurements["present"] = True
    measurements["volume_ml"] = np.sum(mask) * ml_per_voxel

    hu_region = ct_data[mask]
    for func in [np.mean, np.std, np.min, np.median, np.max]:
        measurements[
            f"{func.__name__.replace('amin', 'min').replace('amax', 'max')}_hu"
        ] = float(
            func(hu_region)  # type: ignore
        )
    for p in [25, 75]:
        measurements[f"{p}th_percentile_hu"] = float(np.percentile(hu_region, p))
    if autochton_mean is not None and autochton_std is not None:
        measurements["cnr"] = (np.mean(hu_region) - autochton_mean) / autochton_std
    else:
        measurements["cnr"] = None

    return measurements


def compute_lung_measurement(
    ct_data: np.ndarray,
    region_data: np.ndarray,
    ids: List[int],
    autochton_mean: Optional[float],
    autochton_std: Optional[float],
    img_spacing: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mask = create_mask(region_data, ids)
    fat_mask = np.logical_and(
        mask,
        np.logical_and(
            np.greater_equal(ct_data, -200),
            np.less_equal(ct_data, -40),
        ),
    )
    return fat_mask, metrics_for_region(
        ct_data=ct_data,
        mask=fat_mask,
        autochton_mean=autochton_mean,
        autochton_std=autochton_std,
        img_spacing=img_spacing,
    )


def pulmonary_fat(
    ct_data: np.ndarray,
    region_image: sitk.Image,
    label_map: Dict[str, int],
    autochton_mean: Optional[float],
    autochton_std: Optional[float],
    segmentation_folder: Path,
) -> Dict[str, Any]:
    measurements = {}
    lung_masks = [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
    ]
    region_data = sitk.GetArrayFromImage(region_image)
    for region_name in lung_masks:
        _, measurements["pulmonary_fat_" + region_name] = compute_lung_measurement(
            ct_data=ct_data,
            region_data=region_data,
            ids=[label_map[region_name]],
            autochton_mean=autochton_mean,
            autochton_std=autochton_std,
            img_spacing=region_image.GetSpacing(),  # type: ignore
        )
    for side in ["left", "right"]:
        parts = [ll for ll in lung_masks if ll.endswith(side)]
        _, measurements[f"pulmonary_fat_lobe_{side}"] = compute_lung_measurement(
            ct_data=ct_data,
            region_data=region_data,
            ids=[label_map[ll] for ll in parts],
            autochton_mean=autochton_mean,
            autochton_std=autochton_std,
            img_spacing=region_image.GetSpacing(),  # type: ignore
        )

    fat_mask, measurements["pulmonary_fat_lungs"] = compute_lung_measurement(
        ct_data=ct_data,
        region_data=region_data,
        ids=[label_map[ll] for ll in lung_masks],
        autochton_mean=autochton_mean,
        autochton_std=autochton_std,
        img_spacing=region_image.GetSpacing(),  # type: ignore
    )
    mask_img = sitk.GetImageFromArray(fat_mask.astype(np.uint8))
    mask_img.CopyInformation(region_image)  # type: ignore
    sitk.WriteImage(mask_img, str(segmentation_folder / "pulmonary_fat.nii.gz"), True)

    return measurements


def metrics_for_each_region(
    ct_data: np.ndarray,
    region_data: np.ndarray,
    label_map: Dict[str, int],
    autochton_mean: Optional[float],
    autochton_std: Optional[float],
    img_spacing: np.ndarray,
) -> Dict[str, Any]:
    measurements = {}
    for region_name in label_map:
        mask = create_mask(region_data, label_map[region_name])
        measurements[region_name] = metrics_for_region(
            ct_data=ct_data,
            mask=mask,
            autochton_mean=autochton_mean,
            autochton_std=autochton_std,
            img_spacing=img_spacing,
        )
    return measurements


def compute_measurements(
    ct_path: Path,
    segmentation_folder: Path,
    models: List[str],
) -> Dict[str, Any]:
    measurements: Dict[str, Any] = {
        "segmentations": {},
        "info": {},
    }
    if len(models) == 0:
        return measurements
    logger.info(f"Computing measurements for the computed segmentations: {models}")
    ct_image = sitk.ReadImage(str(ct_path))
    ct_data = sitk.GetArrayViewFromImage(ct_image)
    autochton_mean, autochton_std = None, None
    for model_name in models:
        if model_name == "total":
            model_path = segmentation_folder / "total.nii.gz"
        else:
            model_path = (
                segmentation_folder
                / f"{ADDITIONAL_MODELS_OUTPUT_NAME[model_name]}.nii.gz"
            )
        if not model_path.exists():
            continue
        model_image = sitk.ReadImage(str(model_path))
        assert np.isclose(
            ct_image.GetSpacing(), model_image.GetSpacing()  # type: ignore
        ).all(), "The spacing of the image and of the segmentation should be the same"
        model_data = sitk.GetArrayViewFromImage(model_image)
        if model_name == "total":
            autochton_mean, autochton_std = autochton_reference(
                ct_data=ct_data,
                autochthon_right_mask=create_mask(
                    model_data, reverse_class_map_complete["total_autochthon_right"]
                ),
                autochthon_left_mask=create_mask(
                    model_data, reverse_class_map_complete["total_autochthon_left"]
                ),
            )
        label_map = {
            k[len(model_name) + 1 :]: v
            for k, v in reverse_class_map_complete.items()
            if k.startswith(model_name) and not k.startswith(model_name + "_v2")
        }
        measurements["segmentations"][model_name] = metrics_for_each_region(
            ct_data=ct_data,
            region_data=model_data,
            label_map=label_map,
            autochton_mean=autochton_mean,
            autochton_std=autochton_std,
            img_spacing=ct_image.GetSpacing(),  # type: ignore
        )
        if model_name == "total":
            measurements["segmentations"][model_name] = {
                **measurements["segmentations"][model_name],
                **pulmonary_fat(
                    ct_data=ct_data,
                    region_image=model_image,
                    label_map=label_map,
                    autochton_mean=autochton_mean,
                    autochton_std=autochton_std,
                    segmentation_folder=segmentation_folder,
                ),
            }

    measurements["info"]["autochton_mean"] = autochton_mean
    measurements["info"]["autochton_std"] = autochton_std

    return measurements
