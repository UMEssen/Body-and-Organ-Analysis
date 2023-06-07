import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import SimpleITK as sitk
from body_composition_analysis.io import load_image
from totalsegmentator.util import create_mask, ADDITIONAL_MODELS_OUTPUT_NAME
from totalsegmentator.map_to_binary import reverse_class_map_complete

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


def metrics_for_each_region(
    ct_data: np.ndarray,
    region_data: np.ndarray,
    label_map: Dict[str, int],
    autochton_mean: Optional[float],
    autochton_std: Optional[float],
    img_spacing: np.ndarray,
) -> Dict[str, Any]:
    ml_per_voxel = np.prod(img_spacing) / 1000.0
    measurements = {}
    for region_name in label_map:
        measurements[region_name] = {}
        mask = create_mask(region_data, label_map[region_name])
        if np.sum(mask) == 0:
            measurements[region_name]["present"] = False
            continue

        measurements[region_name]["present"] = True
        measurements[region_name]["volume_ml"] = np.sum(mask) * ml_per_voxel

        hu_region = ct_data[mask]
        for func in [np.mean, np.std, np.min, np.median, np.max]:
            measurements[region_name][
                f"{func.__name__.replace('amin', 'min').replace('amax', 'max')}_hu"
            ] = float(
                func(hu_region)  # type: ignore
            )
        for p in [25, 75]:
            measurements[region_name][f"{p}th_percentile_hu"] = float(
                np.percentile(hu_region, p)
            )
        if autochton_mean is not None and autochton_std is not None:
            measurements[region_name]["cnr"] = (
                np.mean(hu_region) - autochton_mean
            ) / autochton_std
        else:
            measurements[region_name]["cnr"] = None
    return measurements


def compute_measurements(
    ct_path: Path,
    segmentation_folder: Path,
    models: List[str],
) -> Dict[str, Any]:
    measurements = {"segmentations": {}, "info": {}}
    if len(models) == 0:
        return measurements
    logger.info(f"Computing measurements for the computed segmentations: {models}")
    ct_image = load_image(ct_path)
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
        model_image = load_image(model_path)
        assert np.isclose(
            ct_image.GetSpacing(), model_image.GetSpacing()
        ).all(), (  # type: ignore
            "The spacing of the image and of the segmentation should be the same"
        )
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
        measurements["segmentations"][model_name] = metrics_for_each_region(
            ct_data=ct_data,
            region_data=model_data,
            label_map={
                k[len(model_name) + 1 :]: v
                for k, v in reverse_class_map_complete.items()
                if k.startswith(model_name) and not k.startswith(model_name + "_v2")
            },
            autochton_mean=autochton_mean,
            autochton_std=autochton_std,
            img_spacing=ct_image.GetSpacing(),  # type: ignore
        )

    measurements["info"]["autochton_mean"] = autochton_mean
    measurements["info"]["autochton_std"] = autochton_std

    return measurements
