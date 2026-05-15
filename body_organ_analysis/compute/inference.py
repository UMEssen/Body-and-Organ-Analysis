import json
import logging
import pathlib
from collections.abc import Iterable
from typing import Any

import nibabel as nib
import numpy as np
from body_composition_analysis.commands import run_pipeline
from body_composition_analysis.infer.infer import inference
from body_composition_analysis.io import load_nibabel_image_with_axcodes
from totalsegmentator.python_api import totalsegmentator

from body_organ_analysis.compute.measurements import compute_measurements
from body_organ_analysis.compute.util import convert_resampling_slices

logger = logging.getLogger(__name__)


def range_warning(ct_image_data: np.ndarray) -> None:
    if np.any(ct_image_data < -1024) or np.any(ct_image_data > 3071):
        logger.warning(
            "Unexpected CT values found in input image: "
            "got %s-%s, expected -1024-3071. "
            "The values have been clipped to the expected range. "
            "Please check the segmentations to ensure that everything is correct.",
            np.min(ct_image_data),
            np.max(ct_image_data),
        )


def print_and_collect_image_info(
    ct_path: pathlib.Path,
) -> tuple[np.ndarray, np.ndarray]:
    ct_orig = nib.load(ct_path)
    if ct_orig.ndim != 3:
        raise ValueError(f"Only 3D CT scans are supported not {ct_orig.ndim}D.")
    logger.info("Input image:   %s", ct_path)
    logger.info("Image size:    %s", ct_orig.header.get_data_shape())
    logger.info("Image dtype:   %s", ct_orig.header.get_data_dtype())
    logger.info("Voxel spacing: %s", ct_orig.header.get_zooms())
    logger.info("Input Axcodes: %s", nib.aff2axcodes(ct_orig.affine))
    ct_image_data = load_nibabel_image_with_axcodes(ct_orig, axcodes="LPS")
    range_warning(ct_image_data.get_fdata())

    return ct_image_data.shape, ct_image_data.header.get_zooms()


def compute_all_models(
    ct_path: pathlib.Path,
    segmentation_folder: pathlib.Path,
    models_to_compute: Iterable[str] | str,
    totalsegmentator_params: dict[str, Any],
    bca_params: dict[str, Any] | None = None,
    force_split_threshold: int = 400,
    recompute: bool = True,
    cnr_adjustment: bool = True,
) -> dict[str, int]:
    totalsegmentator_params = totalsegmentator_params or {}
    totalsegmentator_params = totalsegmentator_params.copy()
    bca_params = bca_params or {}
    with_preview = totalsegmentator_params.get("preview", False)
    if "preview" in totalsegmentator_params:
        del totalsegmentator_params["preview"]

    shape, spacing = print_and_collect_image_info(ct_path)
    measurement_models = [
        m for m in models_to_compute if m not in {"bca", "body_parts"}
    ]
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
        logger.info("Computing model %s...", chosen_task)
        seg_file = segmentation_folder / f"{chosen_task}.nii.gz"
        if not recompute and seg_file.is_file():
            logger.info("The model was already computed, skipping...")
            continue
        totalsegmentator(
            input=ct_path,
            task=chosen_task,
            output=seg_file,
            preview=with_preview and chosen_task == "total",
            **totalsegmentator_params,
        )

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
    else:
        logger.info("The total measurements were already computed, skipping...")

    if "bca" in models_to_compute:
        resampling_bca = convert_resampling_slices(
            slices=shape[-1],
            current_sampling=spacing[-1],
            target_resampling=5.0,
        )
        if resampling_bca > force_split_threshold:
            logger.info(
                (
                    "Splitting the image into parts as the "
                    "number of slices %s is more than %s"
                ),
                resampling_bca,
                force_split_threshold,
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
    elif "body_parts" in models_to_compute:
        resampling_bca = convert_resampling_slices(
            slices=shape[-1],
            current_sampling=spacing[-1],
            target_resampling=5.0,
        )
        if resampling_bca > force_split_threshold:
            logger.info(
                (
                    "Splitting the image into parts as the "
                    "number of slices %s is more than %s"
                ),
                resampling_bca,
                force_split_threshold,
            )
        inference(
            ct_path=ct_path,
            output_dir=segmentation_folder,
            task_name="body_parts",
            recompute=recompute,
            force_split=resampling_bca > force_split_threshold,
            totalsegmentator_params=totalsegmentator_params,
        )
    return stats
