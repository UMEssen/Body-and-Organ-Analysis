#!/usr/bin/env python
import sys
import os
import argparse
from pkg_resources import require
from pathlib import Path
import time

import numpy as np
import nibabel as nib
import torch

from totalsegmentator.libs import (
    setup_nnunet,
    download_pretrained_weights,
    combine_masks,
    get_parts_for_regions,
)
from totalsegmentator.preview import generate_preview
from totalsegmentator.statistics import (
    get_basic_statistics_for_entire_dir,
    get_radiomics_features_for_entire_dir,
)
import logging
from totalsegmentator.map_to_binary import class_map


def main():
    parser = argparse.ArgumentParser(
        description="Segment 104 anatomical structures in CT images.",
        epilog="Written by Jakob Wasserthal. If you use this tool please cite https://arxiv.org/abs/2208.05868",
    )

    parser.add_argument(
        "-i",
        metavar="filepath",
        dest="input",
        help="CT nifti image",
        type=lambda p: Path(p).absolute(),
        required=True,
    )

    parser.add_argument(
        "--previous_output",
        metavar="filepath",
        help="CT total seg nifti image",
        type=lambda p: Path(p).absolute(),
    )

    parser.add_argument(
        "-o",
        metavar="directory",
        dest="output",
        help="Output directory for segmentation masks",
        type=lambda p: Path(p).absolute(),
        required=True,
    )

    parser.add_argument(
        "-ml",
        "--ml",
        action="store_true",
        help="Save one multilabel image for all classes",
        default=False,
    )

    parser.add_argument(
        "-nr",
        "--nr_thr_resamp",
        type=int,
        help="Nr of threads for resampling",
        default=1,
    )

    parser.add_argument(
        "-ns",
        "--nr_thr_saving",
        type=int,
        help="Nr of threads for saving segmentations",
        default=6,
    )

    parser.add_argument(
        "-f",
        "--fast",
        action="store_true",
        help="Run faster lower resolution model",
        default=False,
    )

    parser.add_argument(
        "-t",
        "--nora_tag",
        type=str,
        help="tag in nora as mask. Pass nora project id as argument.",
        default="None",
    )

    parser.add_argument(
        "-p",
        "--preview",
        action="store_true",
        help="Generate a png preview of segmentation",
        default=False,
    )

    # cerebral_bleed: Intracerebral hemorrhage
    # liver_vessels: hepatic vessels
    parser.add_argument(
        "-ta",
        "--task",
        choices=[
            "total",
            "lung_vessels",
            "cerebral_bleed",
            "hip_implant",
            "coronary_arteries",
            "body",
            "pleural_pericard_effusion",
            "liver_vessels",
            "test",
        ],
        help="Select which model to use. This determines what is predicted.",
        default="total",
    )

    # todo: for 15mm model only run the models which are needed for these rois
    parser.add_argument(
        "-rs",
        "--roi_subset",
        type=str,
        nargs="+",
        help="Define a subset of classes to save (space separated list of class names)",
    )

    parser.add_argument(
        "-s",
        "--statistics",
        action="store_true",
        help="Calc volume (in mm3) and mean intensity. Results will be in statistics.json",
        default=False,
    )

    parser.add_argument(
        "-r",
        "--radiomics",
        action="store_true",
        help="Calc radiomics features. Requires pyradiomics. Results will be in statistics_radiomics.json",
        default=False,
    )

    parser.add_argument(
        "-cp",
        "--crop_path",
        help="Custom path to masks used for cropping. If not set will use output directory.",
        type=lambda p: Path(p).absolute(),
        default=None,
    )

    parser.add_argument(
        "-bs",
        "--body_seg",
        action="store_true",
        help="Do initial rough body segmentation and crop image to body region",
        default=False,
    )

    parser.add_argument(
        "-fs",
        "--force_split",
        action="store_true",
        help="Process image in 3 chunks for less memory consumption",
        default=False,
    )

    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Print no intermediate outputs",
        default=False,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show more intermediate output",
        default=False,
    )

    parser.add_argument(
        "--test",
        metavar="0|1|2|3",
        choices=[0, 1, 2, 3],
        type=int,
        help="Only needed for unittesting.",
        default=0,
    )

    args = parser.parse_args()

    quiet, verbose = args.quiet, args.verbose
    logging.basicConfig()

    if not quiet:
        print(
            "\nIf you use this tool please cite: https://doi.org/10.48550/arXiv.2208.05868\n"
        )
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    if not torch.cuda.is_available():
        if "nnUNet_USE_TRITON" in os.environ and os.environ["nnUNet_USE_TRITON"] == "1":
            print(
                "No GPU detected, but triton config detected. "
                "This will be slower than running this with a normal GPU."
            )
        else:

            print(
                "No GPU detected. Running on CPU. This can be very slow. "
                "The '--fast' option can help to some extend."
            )

    setup_nnunet()

    from totalsegmentator.nnunet import (
        nnUNet_predict_image,
    )  # this has to be after setting new env vars

    crop_addon = [3, 3, 3]  # default value

    if args.task == "total":
        if args.fast:
            task_id = 256
            resample = 3.0
            trainer = "nnUNetTrainerV2_ep8000_nomirror"
            crop = None
            if not quiet:
                print("Using 'fast' option: resampling to lower resolution (3mm)")
        else:
            task_id = [251, 252, 253, 254, 255]
            resample = 1.5
            trainer = "nnUNetTrainerV2_ep4000_nomirror"
            crop = None
        model = "3d_fullres"
        folds = [0]
    elif args.task == "lung_vessels":
        task_id = 258
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "lung"
        if args.ml:
            raise ValueError(
                "task lung_vessels does not work with option --ml, because of postprocessing."
            )
        if args.fast:
            raise ValueError("task lung_vessels does not work with option --fast")
        model = "3d_fullres"
        folds = [0]
    elif args.task == "covid":
        task_id = 201
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "lung"
        model = "3d_fullres"
        folds = [0]
        print(
            "WARNING: The COVID model finds many types of lung opacity not only COVID. Use with care!"
        )
        if args.fast:
            raise ValueError("task covid does not work with option --fast")
    elif args.task == "cerebral_bleed":
        task_id = 150
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "brain"
        model = "3d_fullres"
        folds = [0]
        if args.fast:
            raise ValueError("task cerebral_bleed does not work with option --fast")
    elif args.task == "hip_implant":
        task_id = 260
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "pelvis"
        model = "3d_fullres"
        folds = [0]
        if args.fast:
            raise ValueError("task hip_implant does not work with option --fast")
    elif args.task == "coronary_arteries":
        task_id = 503
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "heart"
        model = "3d_fullres"
        folds = [0]
        print(
            "WARNING: The coronary artery model does not work very robustly. Use with care!"
        )
        if args.fast:
            raise ValueError("task coronary_arteries does not work with option --fast")
    elif args.task == "body":
        if args.fast:
            task_id = 269
            resample = 6.0
            trainer = "nnUNetTrainerV2"
            crop = None
            model = "3d_fullres"
            folds = [0]
            if not quiet:
                print("Using 'fast' option: resampling to lower resolution (6mm)")
        else:
            task_id = 273
            resample = 1.5
            trainer = "nnUNetTrainerV2"
            crop = None
            model = "3d_fullres"
            folds = [0]
        if args.ml:
            raise ValueError(
                "task body does not work with option --ml, because of postprocessing."
            )
    elif args.task == "pleural_pericard_effusion":
        task_id = 315
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "lung"
        crop_addon = [50, 50, 50]
        model = "3d_fullres"
        folds = None
        if args.fast:
            raise ValueError(
                "task pleural_pericard_effusion does not work with option --fast"
            )
    elif args.task == "liver_vessels":
        task_id = 8
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "liver"
        crop_addon = [20, 20, 20]
        model = "3d_fullres"
        folds = None
        if args.fast:
            raise ValueError("task liver_vessels does not work with option --fast")
    elif args.task == "test":
        task_id = [517]
        resample = None
        trainer = "nnUNetTrainerV2"
        crop = "body"
        model = "3d_fullres"
        folds = [0]

    crop_path = args.output if args.crop_path is None else args.crop_path

    if type(task_id) is list:
        for tid in task_id:
            download_pretrained_weights(tid)
    else:
        download_pretrained_weights(task_id)

    # Generate rough body segmentation (speedup for big images; not useful in combination with --fast option)
    if args.task == "total" and args.body_seg:
        download_pretrained_weights(269)
        st = time.time()
        if not quiet:
            print("Generating rough body segmentation...")
        body_seg = nnUNet_predict_image(
            file_in=args.input,
            file_out=None,
            task_id=269,
            model="3d_fullres",
            folds=[0],
            trainer="nnUNetTrainerV2",
            tta=False,
            multilabel_image=True,
            resample=6.0,
            crop=None,
            crop_path=None,
            task_name="body",
            nora_tag="None",
            preview=False,
            save_binary=True,
            nr_threads_resampling=args.nr_thr_resamp,
            nr_threads_saving=1,
            crop_addon=crop_addon,
            quiet=quiet,
            verbose=verbose,
            test=0,
        )
        crop = body_seg
        if verbose:
            print(f"Rough body segmentation generated in {time.time() - st:.2f}s")

    if args.previous_output is not None:
        to_crop = nib.load(args.previous_output).get_fdata()
        region_names = get_parts_for_regions(crop)
        region_ids = [k for k, v in class_map["total"].items() if v in region_names]
        mask = np.zeros(to_crop.shape, dtype=np.uint8)
        mask[np.isin(to_crop, region_ids)] = 1
        crop = mask

    folds = [0]  # None
    seg = nnUNet_predict_image(
        file_in=args.input,
        file_out=args.output,
        task_id=task_id,
        model=model,
        folds=folds,
        trainer=trainer,
        tta=False,
        multilabel_image=args.ml,
        resample=resample,
        crop=crop,
        crop_path=crop_path,
        task_name=args.task,
        nora_tag=args.nora_tag,
        preview=args.preview,
        nr_threads_resampling=args.nr_thr_resamp,
        nr_threads_saving=args.nr_thr_saving,
        force_split=args.force_split,
        crop_addon=crop_addon,
        roi_subset=args.roi_subset,
        quiet=quiet,
        verbose=verbose,
        test=args.test,
    )
    seg = seg.get_fdata().astype(np.uint8)

    if args.statistics:
        if not quiet:
            print("Calculating statistics...")
        st = time.time()
        stats_dir = args.output.parent if args.ml else args.output
        get_basic_statistics_for_entire_dir(
            seg, args.input, stats_dir / "statistics.json", quiet
        )
        # get_radiomics_features_for_entire_dir(args.input, args.output, args.output / "statistics_radiomics.json")
        if not quiet:
            print(f"  calculated in {time.time() - st:.2f}s")

    if args.radiomics:
        if not quiet:
            print("Calculating radiomics...")
        st = time.time()
        stats_dir = args.output.parent if args.ml else args.output
        get_radiomics_features_for_entire_dir(
            args.input, args.output, stats_dir / "statistics_radiomics.json"
        )
        if not quiet:
            print(f"  calculated in {time.time() - st:.2f}s")


if __name__ == "__main__":
    main()
