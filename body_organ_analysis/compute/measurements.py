import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
from skimage.morphology import binary_erosion
from totalsegmentator.map_to_binary import reverse_class_map_complete

from body_organ_analyzer.compute.util import ADDITIONAL_MODELS_OUTPUT_NAME, create_mask

logger = logging.getLogger(__name__)

ADIPOSE_TISSUE = (-200, -40)


def get_region_minus_fat(
    ct_data: np.ndarray,
    region_mask: np.ndarray,
) -> np.ndarray:
    return np.logical_and(
        region_mask,
        np.logical_or(
            np.less(ct_data, ADIPOSE_TISSUE[0]),
            np.greater(ct_data, ADIPOSE_TISSUE[1]),
        ),
    )


def autochthon_reference(
    ct_data: np.ndarray,
    autochthon_right_mask: np.ndarray,
    autochthon_left_mask: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    autochthon_minus_fat_mask = get_region_minus_fat(
        ct_data=ct_data,
        region_mask=np.logical_or(autochthon_right_mask, autochthon_left_mask),
    )
    autochthon_minus_fat_mask = erode_region(
        autochthon_minus_fat_mask,
    )
    if np.sum(autochthon_minus_fat_mask) == 0:
        return None, None
    return float(np.mean(ct_data[autochthon_minus_fat_mask])), float(
        np.std(ct_data[autochthon_minus_fat_mask])
    )


def erode_region(
    mask: np.ndarray,
    kernel_value: int = 6,
) -> np.ndarray:
    kernel = np.ones([kernel_value] * 3, dtype=np.uint8)
    shrunken_volume = binary_erosion(mask, kernel)

    return shrunken_volume


def metrics_for_region(
    ct_data: np.ndarray,
    mask: np.ndarray,
    autochthon_mean: Optional[float],
    autochthon_std: Optional[float],
    img_spacing: np.ndarray,
    cnr_adjustment: bool = False,
    region_name: str = "",
) -> Dict[str, Any]:
    measurements: Dict[str, Any] = {}
    if np.sum(mask) == 0:
        measurements["present"] = False
        return measurements
    # TODO: If the other values work better, make the "cnr_adjustment" be the default
    #  in the default sheet for autochthon, ivc, pulmonary artery and aorta.
    #  In such case, we should ideally also add a comment column saying that these
    #  measurements are different from the others.
    if cnr_adjustment:
        if "autochthon" in region_name:
            mask = get_region_minus_fat(
                ct_data=ct_data,
                region_mask=mask,
            )
        mask = erode_region(
            mask,
        )
    if np.sum(mask) == 0:
        measurements["present"] = False
        return measurements
    ml_per_voxel = np.prod(img_spacing) / 1000.0
    measurements["present"] = True
    hu_region = ct_data[mask]
    measurements["volume_ml"] = np.sum(mask) * ml_per_voxel
    for func in [np.mean, np.std, np.min, np.median, np.max]:
        measurements[
            f"{func.__name__.replace('amin', 'min').replace('amax', 'max')}_hu"
        ] = float(
            func(hu_region)  # type: ignore
        )
    for p in [25, 75]:
        measurements[f"{p}th_percentile_hu"] = float(np.percentile(hu_region, p))
    if autochthon_mean is not None and autochthon_std is not None:
        measurements["cnr"] = (np.mean(hu_region) - autochthon_mean) / autochthon_std
    else:
        measurements["cnr"] = None

    return measurements


def compute_lung_measurement(
    ct_data: np.ndarray,
    region_data: np.ndarray,
    ids: List[int],
    autochthon_mean: Optional[float],
    autochthon_std: Optional[float],
    img_spacing: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    mask = create_mask(region_data, ids)
    fat_mask = np.logical_and(
        mask,
        np.logical_and(
            np.greater_equal(ct_data, ADIPOSE_TISSUE[0]),
            np.less_equal(ct_data, ADIPOSE_TISSUE[1]),
        ),
    )
    return fat_mask, metrics_for_region(
        ct_data=ct_data,
        mask=fat_mask,
        autochthon_mean=autochthon_mean,
        autochthon_std=autochthon_std,
        img_spacing=img_spacing,
    )


def pulmonary_fat(
    ct_data: np.ndarray,
    region_image: sitk.Image,
    label_map: Dict[str, int],
    autochthon_mean: Optional[float],
    autochthon_std: Optional[float],
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
            autochthon_mean=autochthon_mean,
            autochthon_std=autochthon_std,
            img_spacing=region_image.GetSpacing(),  # type: ignore
        )
    for side in ["left", "right"]:
        parts = [ll for ll in lung_masks if ll.endswith(side)]
        _, measurements[f"pulmonary_fat_lobe_{side}"] = compute_lung_measurement(
            ct_data=ct_data,
            region_data=region_data,
            ids=[label_map[ll] for ll in parts],
            autochthon_mean=autochthon_mean,
            autochthon_std=autochthon_std,
            img_spacing=region_image.GetSpacing(),  # type: ignore
        )

    fat_mask, measurements["pulmonary_fat_lungs"] = compute_lung_measurement(
        ct_data=ct_data,
        region_data=region_data,
        ids=[label_map[ll] for ll in lung_masks],
        autochthon_mean=autochthon_mean,
        autochthon_std=autochthon_std,
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
    autochthon_mean: Optional[float],
    autochthon_std: Optional[float],
    img_spacing: np.ndarray,
    cnr_adjustment: bool = False,
) -> Dict[str, Any]:
    measurements = {}
    for region_name in label_map:
        mask = create_mask(region_data, label_map[region_name])
        measurements[region_name] = metrics_for_region(
            ct_data=ct_data,
            mask=mask,
            autochthon_mean=autochthon_mean,
            autochthon_std=autochthon_std,
            img_spacing=img_spacing,
            cnr_adjustment=cnr_adjustment,
            region_name=region_name,
        )
    if "autochthon_left" in label_map and "autochthon_right" in label_map:
        mask = create_mask(
            region_data,
            [
                label_map["autochthon_left"],
                label_map["autochthon_right"],
            ],
        )
        measurements["autochthon"] = metrics_for_region(
            ct_data=ct_data,
            mask=mask,
            autochthon_mean=autochthon_mean,
            autochthon_std=autochthon_std,
            img_spacing=img_spacing,
            cnr_adjustment=cnr_adjustment,
            region_name="autochthon",
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
    autochthon_mean, autochthon_std = None, None
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
            autochthon_mean, autochthon_std = autochthon_reference(
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
            autochthon_mean=autochthon_mean,
            autochthon_std=autochthon_std,
            img_spacing=ct_image.GetSpacing(),  # type: ignore
        )
        if model_name == "total":
            measurements["segmentations"][model_name] = {
                **measurements["segmentations"][model_name],
                **pulmonary_fat(
                    ct_data=ct_data,
                    region_image=model_image,
                    label_map=label_map,
                    autochthon_mean=autochthon_mean,
                    autochthon_std=autochthon_std,
                    segmentation_folder=segmentation_folder,
                ),
            }
            measurements["cnr_adjusted"] = metrics_for_each_region(
                ct_data=ct_data,
                region_data=model_data,
                label_map={
                    region: value
                    for region, value in label_map.items()
                    if region
                    in {
                        "aorta",
                        "inferior_vena_cava",
                        "pulmonary_artery",
                        "autochthon_left",
                        "autochthon_right",
                    }
                },
                autochthon_mean=autochthon_mean,
                autochthon_std=autochthon_std,
                img_spacing=ct_image.GetSpacing(),  # type: ignore
                cnr_adjustment=True,
            )

    measurements["info"]["autochthon_mean"] = autochthon_mean
    measurements["info"]["autochthon_std"] = autochthon_std

    return measurements
