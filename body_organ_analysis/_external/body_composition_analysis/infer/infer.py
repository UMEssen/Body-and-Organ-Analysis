from totalsegmentator.config import setup_nnunet

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
from body_composition_analysis.tasks import get_task_info

setup_nnunet()
logger = logging.getLogger(__name__)

BCA_TASKS = {
    "body_regions",
    "body_parts",
}


def inference(
    ct_path: Path,
    output_dir: Path,
    task_name: str,
    force_split: bool = False,
    recompute: bool = False,
    crop: nibabel.Nifti1Image | None = None,
    totalsegmentator_params: dict | None = None,
) -> nibabel.Nifti1Image:
    totalsegmentator_params = totalsegmentator_params or {}
    if task_name not in BCA_TASKS:
        raise ValueError(f"The task name {task_name} does not exist.")
    task_specific_params = get_task_info(task_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{task_name}.nii.gz"
    if not recompute and (output_file).is_file():
        logger.info(f"Loading already computed {task_name}...")
        return nibabel.load(output_file)

    logger.info(
        f"Computing model {task_name} with ID {task_specific_params['task_id']}..."
    )
    task_specific_params["crop"] = crop

    _ = nnUNet_predict_image(
        file_in=ct_path,
        file_out=output_file,
        task_name=task_name,
        force_split=force_split,
        multilabel_image=totalsegmentator_params["ml"],
        # preview=totalsegmentator_params["preview"],  # TODO do i need this?
        nr_threads_resampling=totalsegmentator_params["nr_thr_resamp"],
        nr_threads_saving=totalsegmentator_params["nr_thr_saving"],
        quiet=totalsegmentator_params["quiet"],
        **task_specific_params,
    )

    # TODO use output instead of reloading
    logger.info(f"Computing postprocessing for task {task_name}")
    img = sitk.ReadImage(output_file)
    if task_name == "body_parts":
        sitk_output = postprocess_part_segmentation(img)
    elif task_name == "body_regions":
        sitk_output = postprocess_region_segmentation(img)
    sitk.WriteImage(sitk_output, output_file, True)
    return sitk_to_nib(sitk_output)
