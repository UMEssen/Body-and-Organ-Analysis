import json
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Iterable

import nibabel as nib
import numpy as np
from body_organ_analysis._external.body_composition_analysis.commands import run_pipeline
from body_organ_analysis._external.body_composition_analysis.io import load_nibabel_image_with_axcodes
from totalsegmentator.config import setup_nnunet
from totalsegmentator.python_api import totalsegmentator
from body_organ_analysis.compute.measurements import compute_measurements
from body_organ_analysis.compute.util import convert_resampling_slices, create_mask
from body_organ_analysis._external.body_composition_analysis.io import compress

setup_nnunet()
from totalsegmentator.nnunet import nnUNet_predict_image  # noqa

logger = logging.getLogger(__name__)


def remove_debug_segmentations(
    segmentation_folder: pathlib.Path,
    models_to_compute: List[str],
) -> None:
    to_remove = [
        "bca.nii.gz",
        "body.nii.gz",
        "lung_trachea_bronchia.nii.gz",
        "lung_vessels.nii.gz",
    ]
    if "bca" in models_to_compute:
        to_remove += [
            "body_extremities.nii.gz",
            "body_trunc.nii.gz",
        ]
    for r in to_remove:
        (segmentation_folder / r).unlink(missing_ok=True)


def range_warning(ct_image_data: np.ndarray) -> None:
    if np.any(ct_image_data < -1024) or np.any(ct_image_data > 3071):
        logger.warning(
            f"Unexpected CT values found in input image: "
            f"got {np.min(ct_image_data)}-{np.max(ct_image_data)}, expected -1024-3071. "
            f"The values have been clipped to the expected range. "
            "Please check the segmentations to ensure that everything is correct."
        )


def print_and_collect_image_info(
    ct_path: pathlib.Path,
) -> Tuple[np.ndarray, np.ndarray]:
    ct_orig = nib.load(ct_path)
    assert len(ct_orig.header.get_data_shape()) == 3, "Only 3D CT scans are supported."
    logger.info(f"Input image:   {ct_path}")
    logger.info(f"Image size:    {ct_orig.header.get_data_shape()}")
    logger.info(f"Image dtype:   {ct_orig.header.get_data_dtype()}")
    logger.info(f"Voxel spacing: {ct_orig.header.get_zooms()}")
    logger.info(f"Input Axcodes: {nib.aff2axcodes(ct_orig.affine)}")
    ct_image_data = load_nibabel_image_with_axcodes(ct_orig, axcodes="LPS")
    range_warning(ct_image_data.get_fdata())

    return ct_image_data.shape, ct_image_data.header.get_zooms()


def compute_all_models(
    ct_path: pathlib.Path,
    segmentation_folder: pathlib.Path,
    models_to_compute: Iterable[str] | str,
    totalsegmentator_params: Dict[str, Any],
    bca_params: Optional[Dict[str, Any]] = None,
    force_split_threshold: int = 400,
    recompute: bool = True,
    fast: bool = True,
    cnr_adjustment: bool = True,
) -> Dict[str, int]:
    totalsegmentator_params = totalsegmentator_params or {}
    totalsegmentator_params = totalsegmentator_params.copy()
    bca_params = bca_params or {}
    preview_param = totalsegmentator_params.get("preview", False)
    if "preview" in totalsegmentator_params:
        del totalsegmentator_params["preview"]

    shape, spacing = print_and_collect_image_info(ct_path)
    measurement_models = [m for m in models_to_compute if m not in {"bca", "body_parts"}]
    stats = {
        "num_voxels": shape[0] * shape[1] * shape[2],
        "num_slices": shape[2],
        "num_slices_resampled": convert_resampling_slices(
            slices=shape[-1],
            current_sampling=spacing[-1],
            target_resampling=1.5,
        ),
    }

    for chosen_task in measurement_models:
        logger.info(f"Computing model {chosen_task}...")
        seg_file = segmentation_folder / f"{chosen_task}.nii.gz"
        if not recompute and seg_file.is_file():
            logger.info("The model was already computed, skipping...")
            continue
        totalsegmentator(input=ct_path, task=chosen_task, **totalsegmentator_params)
        compress(seg_file.with_suffix(""))


    # TODO move to the place where the file is read
    measurement_file = segmentation_folder / "total-measurements.json"
    if measurement_models and (recompute or not measurement_file.is_file()):
        json_data = compute_measurements(
            ct_path=ct_path,
            segmentation_folder=segmentation_folder,
            models=measurement_models,
            cnr_adjustment=cnr_adjustment,
        )
        with measurement_file.open("w") as ofile:
            json.dump(json_data, ofile, indent=2)
        del json_data

    if "bca" in models_to_compute:
        resampling_bca = convert_resampling_slices(
            slices=shape[-1],
            current_sampling=spacing[-1],
            target_resampling=5.0,
        )
        if resampling_bca > force_split_threshold:
            logger.info(
                f"Splitting the image into parts as the number of slices "
                f"{resampling_bca} is more than {force_split_threshold}"
            )
        run_pipeline(
            input_image=ct_path,
            output_dir=segmentation_folder,
            force_split=resampling_bca > force_split_threshold,
            crop_body=False,
            recompute=recompute,
            totalsegmentator_params=totalsegmentator_params,
            **bca_params,
        )
    return stats
