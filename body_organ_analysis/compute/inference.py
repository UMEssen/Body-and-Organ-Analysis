import json
import logging
import pathlib
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from body_composition_analysis.commands import run_pipeline
from body_composition_analysis.io import load_nibabel_image_with_axcodes
from totalsegmentator.libs import (
    download_pretrained_weights,
    get_parts_for_regions,
    setup_nnunet,
)
from totalsegmentator.map_to_binary import reverse_class_map_complete
from totalsegmentator.postprocessing import postprocess_lung_vessels
from totalsegmentator.task_info import get_task_info

from body_organ_analysis.compute.measurements import compute_measurements
from body_organ_analysis.compute.util import convert_resampling_slices, create_mask

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
    ct_orig = nib.load(ct_path)  # type: ignore
    assert (
        len(ct_orig.header.get_data_shape()) == 3  # type: ignore
    ), "Only 3D CT scans are supported."
    logger.info(f"Input image:   {ct_path}")
    logger.info(f"Image size:    {ct_orig.header.get_data_shape()}")  # type: ignore
    logger.info(f"Image dtype:   {ct_orig.header.get_data_dtype()}")  # type: ignore
    logger.info(f"Voxel spacing: {ct_orig.header.get_zooms()}")  # type: ignore
    logger.info(f"Input Axcodes:    {nib.aff2axcodes(ct_orig.affine)}")  # type: ignore
    ct_image_data = load_nibabel_image_with_axcodes(ct_orig, axcodes="LPS")
    range_warning(ct_image_data.get_fdata())

    return ct_image_data.shape, ct_image_data.header.get_zooms()


def compute_all_models(
    ct_path: pathlib.Path,
    segmentation_folder: pathlib.Path,
    models_to_compute: List[str],
    force_split_threshold: int = 400,
    totalsegmentator_params: Optional[Dict] = None,
    bca_params: Optional[Dict] = None,
    keep_debug_segmentations: bool = False,
    recompute: bool = True,
) -> Dict[str, int]:
    totalsegmentator_params = totalsegmentator_params or {}
    totalsegmentator_params = totalsegmentator_params.copy()
    bca_params = bca_params or {}
    preview_param = totalsegmentator_params.get("preview", False)
    if "preview" in totalsegmentator_params:
        del totalsegmentator_params["preview"]

    shape, spacing = print_and_collect_image_info(ct_path)

    stats = {
        "num_voxels": shape[0] * shape[1] * shape[2],
        "num_slices": shape[2],
        "num_slices_resampled": convert_resampling_slices(
            slices=shape[-1],
            current_sampling=spacing[-1],
            target_resampling=1.5,
        ),
    }

    for chosen_task in [m for m in models_to_compute if m != "bca"]:
        logger.info(f"Computing segmentations for task {chosen_task}")
        task_specific_params = get_task_info(
            chosen_task,
            fast=False,
            multilabel_image=chosen_task not in {"lung_vessels", "body"},
            quiet=True,
        )
        if task_specific_params["multilabel_image"]:
            seg_output = segmentation_folder / f"{chosen_task}.nii.gz"
            if seg_output.exists() and not recompute:
                logger.info("The segmentation was already computed, skipping...")
                continue
        else:
            seg_output = segmentation_folder
            if not recompute and (
                (
                    chosen_task == "body"
                    and (seg_output / f"{chosen_task}.nii.gz").exists()
                )
                or (
                    chosen_task == "lung_vessels"
                    and (seg_output / "lung_vessels_airways.nii.gz").exists()
                )
            ):
                logger.info("The segmentation was already computed, skipping...")
                continue
        if type(task_specific_params["task_id"]) is list:
            for tid in task_specific_params["task_id"]:
                download_pretrained_weights(tid)
        else:
            download_pretrained_weights(task_specific_params["task_id"])
        if task_specific_params["crop"] is not None:
            assert (
                segmentation_folder / "total.nii.gz"
            ).exists(), "The total segmentation is required to compute the crop!"
            tmp_total_data = nib.load(  # type: ignore
                segmentation_folder / "total.nii.gz"
            ).get_fdata()
            old_crop_name = task_specific_params["crop"]
            mask_ids = [
                reverse_class_map_complete[f"total_{n}"]
                for n in get_parts_for_regions(task_specific_params["crop"])
            ]
            task_specific_params["crop"] = create_mask(
                tmp_total_data,
                mask_ids,
            ).astype(np.uint8)
            del tmp_total_data
            if not task_specific_params["crop"].sum():
                logger.info(
                    f"The segmentation for {chosen_task} could not be computed "
                    f"because the main crop region {old_crop_name} is not present."
                )
                continue
        resampled_slices = convert_resampling_slices(
            slices=shape[-1],
            current_sampling=spacing[-1],
            target_resampling=task_specific_params["resample"],
        )
        split = False
        if chosen_task == "total" and resampled_slices > force_split_threshold:
            split = True
            logger.info(
                f"Splitting the image into parts as the number of slices "
                f"{resampled_slices} is more than {force_split_threshold}"
            )
        _ = nnUNet_predict_image(
            file_in=ct_path,
            file_out=seg_output,
            task_name=chosen_task,
            force_split=split,
            preview=preview_param and chosen_task == "total",
            **task_specific_params,
            **totalsegmentator_params,
        )
        if chosen_task == "lung_vessels":
            postprocess_lung_vessels(segmentation_output=seg_output)

    measurement_models = [m for m in models_to_compute if m not in {"bca", "body"}]
    if len(measurement_models) > 0:
        json_data = compute_measurements(
            ct_path=ct_path,
            segmentation_folder=segmentation_folder,
            models=measurement_models,
        )
        with (segmentation_folder / "total-measurements.json").open("w") as ofile:
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
    if not keep_debug_segmentations:
        remove_debug_segmentations(
            segmentation_folder=segmentation_folder,
            models_to_compute=models_to_compute,
        )

    return stats
