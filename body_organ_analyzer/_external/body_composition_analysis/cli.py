import argparse
import json
import logging
import pathlib

from body_composition_analysis import enable_debug_mode
from body_composition_analysis.commands import (
    CHOSEN_BMD_VERTEBRAE,
    compute_bmd_from_vertebrae,
    compute_segmentation,
    report,
    run_pipeline,
    subclassify,
)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--triton-url",
        default="127.0.0.1:8001",
        help="URL to the Triton inference server via gRPC",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        default=False,
        action="store_true",
        help="Print additional information for debugging purposes",
    )
    parser.add_argument(
        "--debug-dir",
        type=pathlib.Path,
        help="Enables debug mode and stores intermediate files on disk",
    )

    subparsers = parser.add_subparsers(dest="command")

    # Run complete BCA pipeline
    subparser = subparsers.add_parser("run-pipeline")
    subparser.add_argument(
        "--input-image",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the CT image",
    )
    subparser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
    )
    subparser.add_argument(
        "--median-filtering",
        default=False,
        action="store_true",
        help="Apply median filtering before thresholding tissues using Hounsfield Unit ranges",
    )
    subparser.add_argument(
        "--examined-body-region", choices=["abdomen", "neck", "thorax"]
    )
    subparser.add_argument(
        "--no-pdf",
        default=False,
        action="store_true",
        help="Skip the PDF generation and only create a bca-measurements.json file.",
    )

    # Compute
    subparser = subparsers.add_parser("compute-bmd")
    subparser.add_argument(
        "--input-image",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the CT image",
    )
    subparser.add_argument(
        "--input-body-regions",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the result of the body region segmentation",
    )
    exclusive_group = subparser.add_mutually_exclusive_group(required=True)
    exclusive_group.add_argument(
        "--input-vertebrae",
        type=pathlib.Path,
        help="Path to the JSON with information about localized vertebrae",
    )
    exclusive_group.add_argument(
        "--input-measurements",
        type=pathlib.Path,
        help="Path to the JSON with all measurements of the pipeline",
    )
    subparser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
    )

    # Compute body region segmentation
    subparser = subparsers.add_parser("compute-regions")
    subparser.add_argument(
        "--input-image",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the CT image",
    )
    subparser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
    )

    # Compute body region segmentation
    subparser = subparsers.add_parser("compute-parts")
    subparser.add_argument(
        "--input-image",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the CT image",
    )
    subparser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
    )

    # Compute vertebrae localization
    subparser = subparsers.add_parser("compute-vertebrae")
    subparser.add_argument(
        "--input-image",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the CT image",
    )
    subparser.add_argument(
        "--input-body-regions",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the result of the body region segmentation",
    )
    subparser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
    )

    # Build report from body region segmentation and subclassified tissues
    subparser = subparsers.add_parser("report")
    subparser.add_argument(
        "--input-image",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the CT image",
    )
    subparser.add_argument(
        "--input-body-parts",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the result of the body part segmentation",
    )
    subparser.add_argument(
        "--input-body-regions",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the result of the body region segmentation",
    )
    subparser.add_argument(
        "--input-tissues",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the result of tissue subclassification",
    )
    subparser.add_argument(
        "--input-vertebrae",
        type=pathlib.Path,
        help="Path to the JSON with information about localized vertebrae",
    )
    subparser.add_argument(
        "--input-bmd",
        type=pathlib.Path,
        help="Path to the JSON with BMD results",
    )
    subparser.add_argument(
        "--output-report",
        type=pathlib.Path,
    )
    subparser.add_argument(
        "--output-measurements",
        required=True,
        type=pathlib.Path,
    )
    subparser.add_argument(
        "--examined-body-region", choices=["abdomen", "neck", "thorax"]
    )
    subparser.add_argument(
        "--no-pdf",
        default=False,
        action="store_true",
        help="Skip the PDF generation and only create a bca-measurements.json file.",
    )

    # Tissue subclassification from CT image and body region segmentation
    subparser = subparsers.add_parser("subclassify")
    subparser.add_argument(
        "--input-image",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the CT image",
    )
    subparser.add_argument(
        "--input-body-regions",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the result of the body region segmentation",
    )
    subparser.add_argument(
        "--output",
        required=True,
        type=pathlib.Path,
    )
    subparser.add_argument(
        "--median-filtering",
        default=False,
        action="store_true",
        help="Apply median filtering before thresholding tissues using Hounsfield Unit ranges",
    )

    return parser


def run() -> None:
    parser = get_parser()
    args = parser.parse_args()

    if args.debug_dir:
        print(args.debug_dir)
        enable_debug_mode(args.debug_dir)

    logging.basicConfig()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    if args.command is None:
        parser.print_help()
        exit()

    totalsegmentator_params = dict(
        tta=False,
        preview=False,
        nr_threads_resampling=1,
        nr_threads_saving=1,
        nora_tag="None",
        roi_subset=None,
        quiet=False,
        verbose=False,
        test=0,
        crop_path=None,
    )

    if args.command == "compute-bmd":
        if args.input_vertebrae is not None:
            with args.input_vertebrae.open() as ifile:
                info = json.load(ifile)
                vertebrae_slice_mapping = {
                    k: info[k] for k in CHOSEN_BMD_VERTEBRAE if k in info
                }
        else:
            with args.input_measurements.open() as ifile:
                info = json.load(ifile)
                vertebrae_slice_mapping = {
                    vid.upper(): (
                        info["aggregated"][vid]["min_slice_idx"],
                        info["aggregated"][vid]["max_slice_idx"],
                    )
                    for vid in [v.lower() for v in CHOSEN_BMD_VERTEBRAE]
                    if vid in info["aggregated"]
                }
        if len(vertebrae_slice_mapping) == 0:
            raise ValueError("No vertebrae found in measurements JSON")

        compute_bmd_from_vertebrae(
            input_image=args.input_image,
            input_body_regions=args.input_body_regions,
            vertebrae_slice_mapping=vertebrae_slice_mapping,
            output=args.output,
        )
    elif args.command == "compute-regions":
        compute_segmentation(
            input_image=args.input_image,
            output=args.output,
            task_name="bca",
            totalsegmentator_params=totalsegmentator_params,
        )
    elif args.command == "compute-parts":
        compute_segmentation(
            input_image=args.input_image,
            output=args.output,
            task_name="body",
            totalsegmentator_params=totalsegmentator_params,
        )
    elif args.command == "compute-vertebrae":
        compute_segmentation(
            input_image=args.input_image,
            output=args.output,
            task_name="vertebrae",
        )
    elif args.command == "report":
        if not args.no_pdf and args.output_report is None:
            parser.error("PDF report generation requires --output-report")
        report(
            input_image=args.input_image,
            input_body_regions=args.input_body_regions,
            input_body_parts=args.input_body_parts,
            input_tissues=args.input_tissues,
            input_vertebrae=args.input_vertebrae,
            input_bmd=args.input_bmd,
            examined_body_region=args.examined_body_region,
            output_report=args.output_report,
            output_measurements=args.output_measurements,
            save_pdf=not args.no_pdf,
        )
    elif args.command == "subclassify":
        subclassify(
            input_image=args.input_image,
            input_body_regions=args.input_body_regions,
            output=args.output,
            median_filtering=args.median_filtering,
        )
    elif args.command == "run-pipeline":
        run_pipeline(
            input_image=args.input_image,
            output_dir=args.output_dir,
            examined_body_region=args.examined_body_region,
            median_filtering=args.median_filtering,
            save_pdf=not args.no_pdf,
            totalsegmentator_params=totalsegmentator_params,
        )
