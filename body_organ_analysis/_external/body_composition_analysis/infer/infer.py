from totalsegmentator.libs import setup_nnunet

setup_nnunet()

import logging
from pathlib import Path

import nibabel
import SimpleITK as sitk
from body_composition_analysis.body_parts.postprocess import (
    postprocess_part_segmentation,
)
from body_composition_analysis.body_regions.postprocess import (
    postprocess_region_segmentation,
)
from body_composition_analysis.io import sitk_to_nib
from totalsegmentator.libs import download_pretrained_weights
from totalsegmentator.nnunet import nnUNet_predict_image
from totalsegmentator.task_info import get_task_info

logger = logging.getLogger(__name__)

TASK_TO_POST_NAME = {
    "bca": "body-regions",
    "body": "body-parts",
    "vertebrae": "vertebrae",
}


def inference(
    ct_path: Path,
    output_dir: Path,
    task_name: str,
    force_split: bool = False,
    recompute: bool = False,
    postprocess: bool = False,
    crop: nibabel.Nifti1Image = None,
    totalsegmentator_params: dict = None,
) -> nibabel.Nifti1Image:
    totalsegmentator_params = totalsegmentator_params or {}
    if task_name not in TASK_TO_POST_NAME.keys():
        raise ValueError(f"The task name {task_name} does not exist.")
    task_specific_params = get_task_info(
        task_name, fast=False, multilabel_image=task_name in {"bca", "vertebrae"}
    )
    if (
        task_name == "vertebrae"
        and not recompute
        and (output_dir / "total.nii.gz").exists()
    ):
        logger.info("Loading total.nii.gz...")
        return nibabel.load(output_dir / "total.nii.gz")

    if ".nii" not in output_dir.name:
        no_post_path = output_dir / f"{task_name}.nii.gz"
        post_path = output_dir / f"{TASK_TO_POST_NAME[task_name]}.nii.gz"
    else:
        no_post_path = output_dir
        post_path = output_dir

    if postprocess and post_path.exists() and not recompute:
        logger.info(f"Loading already computed {post_path}...")
        return nibabel.load(post_path)

    if not no_post_path.exists() or recompute:
        logger.info(
            f"Computing model {task_name} with ID {task_specific_params['task_id']}..."
        )
        if isinstance(task_specific_params["task_id"], list):
            for t_id in task_specific_params["task_id"]:
                download_pretrained_weights(t_id)
        else:
            download_pretrained_weights(task_specific_params["task_id"])
        task_specific_params["crop"] = crop
        output = nnUNet_predict_image(
            file_in=ct_path,
            file_out=no_post_path
            if task_specific_params["multilabel_image"]
            else no_post_path.parent,
            task_name=task_name if task_name not in {"vertebrae"} else "total",
            force_split=force_split,
            axcodes="LPS" if task_name == "bca" else "RAS",
            **task_specific_params,
            **totalsegmentator_params,
        )
        if not postprocess:
            return output

    if postprocess and task_name in {"body", "bca"}:
        # SimpleITK is much faster at processing
        if task_name == "body":
            logger.info("Computing postprocessing for task body")
            trunc_img = sitk.ReadImage(str(no_post_path.parent / "body_trunc.nii.gz"))
            extr_img = sitk.ReadImage(
                str(no_post_path.parent / "body_extremities.nii.gz")
            )
            sitk_output = postprocess_part_segmentation(trunc_img, extr_img)
        else:
            logger.info("Computing postprocessing for task bca")
            bca_img = sitk.ReadImage(str(no_post_path))
            sitk_output = postprocess_region_segmentation(bca_img)
        sitk.WriteImage(sitk_output, str(post_path), True)
        return sitk_to_nib(sitk_output)
    logger.info(f"Loading already computed {post_path}...")
    return nibabel.load(post_path)
