import argparse
import logging
import os
import pathlib
import time

from totalsegmentator.statistics import get_radiomics_features_for_entire_dir

from body_organ_analysis.commands import analyze_ct

logger = logging.getLogger(__name__)


def get_parser() -> argparse.ArgumentParser:
    # TODO: Some variables are currently not accessible via CLI
    parser = argparse.ArgumentParser()
    # Run complete BCA pipeline
    parser.add_argument(
        "--input-image",
        required=True,
        type=pathlib.Path,
        help="Path to the ITK image which contains the CT image",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=pathlib.Path,
    )

    parser.add_argument(
        "--use-study-prefix",
        default=False,
        action="store_true",
        help="Output files will be prefixed with the study name",
    )

    parser.add_argument(
        "--models",
        required=True,
        default="all",
        help="Plus separated list of models to compute. "
        "If 'all' is specified, all models will be computed",
    )

    parser.add_argument(
        "--skip-contrast-information",
        default=False,
        action="store_true",
        help="Whether to skip the computation of the IV phase and GIT contrast presence.",
    )

    parser.add_argument(
        "-nr",
        "--nr_thr_resamp",
        type=int,
        help="Nr of threads for resampling",
        default=1,
    )

    parser.add_argument(
        "-p",
        "--preview",
        action="store_true",
        help="Generate a png preview of segmentation",
        default=False,
    )

    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Generate all segmentations from scratch, even if they already exist",
        default=False,
    )

    parser.add_argument(
        "-ns",
        "--nr_thr_saving",
        type=int,
        help="Nr of threads for saving segmentations",
        default=6,
    )

    parser.add_argument(
        "-r",
        "--radiomics",
        action="store_true",
        help="Calc radiomics features. Requires pyradiomics. Results will be in statistics_radiomics.json",
        default=False,
    )

    parser.add_argument(
        "--keep-debug-segmentations",
        default=False,
        action="store_true",
        help="Keep all the debug images generated during the segmentation process",
    )

    parser.add_argument(
        "--triton-url",
        default=None,
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
        "--nnunet-verbose",
        default=False,
        action="store_true",
        help="Print all the output logs of nnunet",
    )

    parser.add_argument(
        "--bca-median-filtering",
        default=False,
        action="store_true",
        help="Apply median filtering before thresholding tissues using Hounsfield Unit ranges",
    )
    parser.add_argument(
        "--bca-examined-body-region", choices=["abdomen", "neck", "thorax"]
    )
    parser.add_argument(
        "--bca-no-pdf",
        default=False,
        action="store_true",
        help="Skip the PDF generation and only create a bca-measurements.json file.",
    )

    return parser


def run() -> None:
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig()
    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    if args.triton_url is not None:
        os.environ["TRITON_URL"] = args.triton_url
        os.environ["nnUNet_USE_TRITON"] = "1"
    else:
        os.environ["nnUNet_USE_TRITON"] = "0"

    if args.models == "all":
        models_to_compute = [
            "body",
            "total",
            "lung_vessels",
            "cerebral_bleed",
            "hip_implant",
            "coronary_arteries",
            "pleural_pericard_effusion",
            "liver_vessels",
            "bca",
        ]
    else:
        models_to_compute = args.models.split("+")

    analyze_ct(
        input_folder=args.input_image,
        processed_output_folder=args.output_dir,
        excel_output_folder=args.output_dir,
        models=models_to_compute,
        compute_contrast_information=not args.skip_contrast_information,
        total_preview=args.preview,
        nr_thr_resamp=args.nr_thr_resamp,
        nr_thr_saving=args.nr_thr_saving,
        bca_median_filtering=args.bca_median_filtering,
        bca_examined_body_region=args.bca_examined_body_region,
        bca_pdf=not args.bca_no_pdf,
        bca_compute_bmd=False,
        recompute=args.force_recompute,
        keep_debug_information=args.keep_debug_segmentations,
        nnunet_verbose=args.nnunet_verbose,
    )

    if args.radiomics:
        logger.info("Calculating radiomics...")
        st = time.time()
        get_radiomics_features_for_entire_dir(
            args.input_image,
            args.output_dir,
            args.output_dir / "statistics_radiomics.json",
        )
        logger.info(f"  calculated in {time.time() - st:.2f}s")

    if args.use_study_prefix:
        study_name = args.input_image.name.replace(".nii.gz", "")
        for f in args.output_dir.glob("*"):
            f.rename(f.parent / f"{study_name}_{f.name}")


if __name__ == "__main__":
    run()
