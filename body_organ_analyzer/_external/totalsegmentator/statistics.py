from pathlib import Path
import json
from functools import partial
import time

import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from p_tqdm import p_map
import numpy.ma as ma

from totalsegmentator.map_to_binary import class_map_radiomics
import logging

logger = logging.getLogger(__name__)


def get_radiomics_features(seg_file, img_file="ct.nii.gz"):
    # import SimpleITK as sitk
    import radiomics

    radiomics.logger.setLevel(logging.WARNING)
    from radiomics import featureextractor

    features = {}
    labels = [int(l) for l in np.unique(nib.load(seg_file).get_fdata())]
    task_name = seg_file.name.replace(".nii.gz", "")
    class_map_task = (
        class_map_radiomics[task_name] if task_name in class_map_radiomics else None
    )
    # logger.info(
    #     f"Task: {task_name} labels: {labels}\n"
    #     f"{class_map_task.keys() if class_map_task else None}"
    # )
    if len(labels) > 1:
        settings = {}
        # settings["binWidth"] = 25
        # settings["resampledPixelSpacing"] = None  # [3,3,3] is an example for defining resampling (voxels with size 3x3x3mm)
        settings["resampledPixelSpacing"] = [3, 3, 3]
        # settings["interpolator"] = sitk.sitkBSpline
        settings["geometryTolerance"] = 1e-3  # default: 1e-6
        settings["featureClass"] = ["shape"]
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        # Only use subset of features
        extractor.disableAllFeatures()
        extractor.enableFeatureClassByName("shape")
        extractor.enableFeatureClassByName("firstorder")
        for lab in labels[1:]:
            key = class_map_task[lab] if class_map_task is not None else lab
            features[key] = {}
            try:
                lab_features = extractor.execute(
                    str(img_file), str(seg_file), label=lab
                )
                for k, v in lab_features.items():
                    if k.startswith("original_"):
                        # round to 4 decimals and cast to python float
                        features[key][k.replace("original_", "")] = round(float(v), 4)
            except Exception as e:
                logger.warning(
                    f"WARNING: radiomics raised an exception for task {task_name} and label {lab}"
                    f": {e}"
                )
                features[key]["exception"] = {}
    else:
        logger.warning("WARNING: Entire mask is 0 or 1. Setting all features to 0")
        key = "background" if labels[0] == 0 else class_map_task[labels[0]]
        features[key] = {}

    # only keep subset of features
    # meaningful_features = ['shape_Elongation', 'shape_Flatness', 'shape_LeastAxisLength']
    # features = {k: v for k, v in features.items() if k in meaningful_features}
    return seg_file.name.split(".")[0], features


def get_radiomics_features_for_entire_dir(
    ct_file: Path,
    mask_dir: Path,
    file_out: Path,
):
    masks = sorted(list(mask_dir.glob("*.nii.gz")))
    stats = p_map(
        partial(get_radiomics_features, img_file=ct_file),
        masks,
        num_cpus=1,
        disable=False,
    )
    stats = {mask_name: stats for mask_name, stats in stats}
    with open(file_out, "w") as f:
        json.dump(stats, f, indent=4)


def get_basic_statistics_for_entire_dir(
    seg: np.array, ct_file: Path, file_out: Path, quiet: bool = False
):
    ct_img = nib.load(ct_file)
    ct = ct_img.get_fdata()
    spacing = ct_img.header.get_zooms()
    vox_vol = spacing[0] * spacing[1] * spacing[2]
    stats = {}
    for k, mask_name in tqdm(class_map["total"].items(), disable=quiet):
        stats[mask_name] = {}
        # data = nib.load(mask).get_fdata()  # loading: 0.6s
        data = seg == k  # 0.18s
        stats[mask_name]["volume"] = data.sum() * vox_vol  # vol in mm3; 0.2s
        roi_mask = (data > 0).astype(np.uint8)  # 0.16s
        # stats[mask_name]["intensity"] = ct[roi_mask > 0].mean().round(2) if roi_mask.sum() > 0 else 0.0  # 3.0s
        stats[mask_name]["intensity"] = (
            np.average(ct, weights=roi_mask).round(2) if roi_mask.sum() > 0 else 0.0
        )  # 0.9s

    # For nora json is good
    # For other people csv might be better -> not really because here only for one subject each -> use json
    with open(file_out, "w") as f:
        json.dump(stats, f, indent=4)
