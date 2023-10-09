import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
from body_composition_analysis.io import load_image
from scipy import spatial
from totalsegmentator.map_to_binary import reverse_class_map_complete

from body_organ_analysis.compute.geometry import find_axes
from body_organ_analysis.compute.util import (
    ADDITIONAL_MODELS_OUTPUT_NAME,
    convert_name,
    create_mask,
)

logger = logging.getLogger(__name__)


def major_minor_axis(
    l3_mask: np.ndarray,
    body_mask: np.ndarray,
    img_spacing: np.ndarray,
    plot_axes: Path = None,
) -> Tuple[Optional[float], Optional[float]]:
    if np.sum(l3_mask) == 0 or np.sum(body_mask) == 0:
        return None, None
    slices = np.where(l3_mask.any(axis=(1, 2)))[0]
    # Middle slice
    middle_slice = body_mask[int(np.median(slices)), :, :]
    if np.sum(middle_slice) == 0:
        return None, None
    major_p1, major_p2, minor_p1, minor_p2 = find_axes(middle_slice)
    if plot_axes is not None:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(middle_slice, cmap="gray")
        ax.plot((major_p1.x, major_p2.x), (major_p1.y, major_p2.y), "-g", linewidth=2.5)
        ax.plot((minor_p1.x, minor_p2.x), (minor_p1.y, minor_p2.y), "-b", linewidth=2.5)
        plt.axis("off")
        plt.savefig(plot_axes / "major_minor_axis.png", dpi=200, bbox_inches="tight")
    avg_spacing = np.mean(img_spacing)  # type: ignore
    # Compute the mean axis,
    # multiply by the spacing between the pixels (mm)
    return (
        spatial.distance.euclidean(major_p1.to_list(), major_p2.to_list())
        * avg_spacing,
        spatial.distance.euclidean(minor_p1.to_list(), minor_p2.to_list())
        * avg_spacing,
    )


def get_cnr_for_region(measurements: Dict[str, Any], region: str) -> Any:
    if measurements["segmentations"]["total"][region]["present"]:
        return measurements["segmentations"]["total"][region]["cnr"]
    return None


def compute_segmentator_metrics(
    ct_path: Path,
    segmentation_folder: Path,
    store_axes: bool = False,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    measurements_path = segmentation_folder / "total-measurements.json"
    with measurements_path.open("r") as of:
        json_measurements = json.load(of)

    autochton_std = json_measurements["info"]["autochton_std"]

    cnr_aorta = get_cnr_for_region(json_measurements, "aorta")
    cnr_vci = get_cnr_for_region(json_measurements, "inferior_vena_cava")
    crn_pv = get_cnr_for_region(json_measurements, "portal_vein_and_splenic_vein")

    image_info = load_image(ct_path)
    major_axis, minor_axis, mean_axis = None, None, None
    if (segmentation_folder / "total.nii.gz").exists() and (
        segmentation_folder / "body-parts.nii.gz"
    ).exists():
        region_image = load_image(segmentation_folder / "total.nii.gz")
        region_data = sitk.GetArrayViewFromImage(region_image)
        body_image = load_image(segmentation_folder / "body-parts.nii.gz")
        body_data = sitk.GetArrayViewFromImage(body_image)
        major_axis, minor_axis = major_minor_axis(
            l3_mask=create_mask(
                region_data, reverse_class_map_complete["total_vertebrae_L3"]
            ),
            body_mask=create_mask(body_data, 1),
            img_spacing=image_info.GetSpacing()[:2],  # type: ignore
            plot_axes=segmentation_folder if store_axes else None,
        )
    if major_axis is not None and minor_axis is not None:
        major_axis /= 10
        minor_axis /= 10
        mean_axis = (major_axis + minor_axis) / 2

    records: List[Dict[str, Any]] = []
    for model_name in json_measurements["segmentations"]:
        for region in json_measurements["segmentations"][model_name]:
            base_dict = {
                "ModelName": convert_name(model_name),
                "BodyRegion": convert_name(region),
            }
            for key, val in json_measurements["segmentations"][model_name][
                region
            ].items():
                new_key = convert_name(key)
                if "Hu" in new_key:
                    new_key = new_key.replace("Hu", "HU")
                elif "Cnr" == new_key:
                    new_key = "CNR"
                base_dict[new_key] = val
            records.append(base_dict)

    for model_name, filename in ADDITIONAL_MODELS_OUTPUT_NAME.items():
        model_path = segmentation_folder / f"{filename}.nii.gz"
        if not model_path.exists():
            records.append({"ModelName": convert_name(model_name), "Present": False})
            continue
    additional_info = []
    for name, value in [
        ("Noise", autochton_std),
        ("CNRAorta", cnr_aorta),
        ("CNRVCI", cnr_vci),
        ("CNRPortalSplenicVein", crn_pv),
        ("MaxAxisL3_cm", major_axis),
        ("MinAxisL3_cm", minor_axis),
        ("MeanAxisL3_cm", mean_axis),
    ]:
        if value is not None:
            additional_info.append({"name": name, "value": value})
    return (
        additional_info,
        pd.DataFrame(records).sort_values(by=["ModelName", "BodyRegion"]),
    )
