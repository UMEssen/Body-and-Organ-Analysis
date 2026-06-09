import argparse
import logging
import os
import time
import warnings
from pathlib import Path

from totalsegmentator.config import set_license_number
from totalsegmentator.statistics import get_radiomics_features_for_entire_dir

from body_organ_analysis.commands import analyze_ct
from body_organ_analysis.compute.config import (
    env_bool,
    env_str,
    resolve_device,
    resolve_models,
)
from body_organ_analysis.compute.constants import ALL_MODELS

logger = logging.getLogger(__name__)


def _validate_models(spec: str) -> set[str]:
    """Resolve a '+'-separated model spec for argparse."""
    try:
        return resolve_models(spec, strict=True)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def get_parser() -> argparse.ArgumentParser:
    # TODO: Some variables are currently not accessible via CLI
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-image",
        default="/dicoms",
        type=Path,
        help="Path to the NIfTI file or DICOM directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="/workspace",
        type=Path,
        help="Path to the output files from the BOA calculation",
    )
    parser.add_argument(
        "--use-study-prefix",
        default=False,
        action="store_true",
        help="Output files will be prefixed with the study name",
    )
    parser.add_argument(
        "-m",
        "--models",
        required=True,
        type=_validate_models,
        metavar="MODEL[+MODEL...]",
        help=(
            "Plus-separated list of models to compute, e.g. 'total+bca'. "
            "Use 'all' to compute everything. "
            f"Available: {', '.join(sorted(ALL_MODELS))}."
        ),
    )
    parser.add_argument(
        "--skip-contrast-information",
        default=None,
        action="store_true",
        help=(
            "Whether to skip the computation of the IV phase and GIT contrast presence."
        ),
    )
    parser.add_argument(
        "-nr",
        "--nr_thr_resamp",
        default=1,
        type=int,
        help="Nr of threads for resampling",
    )
    parser.add_argument(
        "-p",
        "--preview",
        default=False,
        action="store_true",
        help="Generate a png preview of segmentation",
    )
    parser.add_argument(
        "--force-recompute",
        default=False,
        action="store_true",
        help="Generate all segmentations from scratch, even if they already exist",
    )
    parser.add_argument(
        "-ns",
        "--nr_thr_saving",
        default=6,
        type=int,
        help="Nr of threads for saving segmentations",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help=(
            "Device used for inference. Accepts 'gpu', 'cuda', 'gpu:<id>', "
            "'cuda:<id>', 'cpu' or 'mps'."
        ),
    )
    parser.add_argument(
        "-r",
        "--radiomics",
        default=False,
        action="store_true",
        help=(
            "Calc radiomics features. Requires pyradiomics. "
            "Results will be in statistics_radiomics.json"
        ),
    )
    parser.add_argument(
        "--cnr-adjustment",
        default=False,
        action="store_true",
        help=(
            "Apply CNR-adjusted measurements for supported TotalSegmentator "
            "regions and add a dedicated cnr-adjusted Excel sheet."
        ),
    )
    parser.add_argument(
        "--triton-url",
        default=None,
        help="URL to the Triton inference server via gRPC",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=None,
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
        help=(
            "Apply median filtering before thresholding "
            "tissues using Hounsfield Unit ranges"
        ),
    )
    parser.add_argument(
        "--bca-examined-body-region",
        choices=["abdomen", "neck", "thorax"],
        help="Limit BCA report measurements to the selected body region.",
    )
    parser.add_argument(
        "--bca-no-pdf",
        default=None,
        action="store_true",
        help="Skip the PDF generation and only create a bca-measurements.json file.",
    )
    parser.add_argument(
        "--fast-bca",
        default=None,
        action="store_true",
        help=("Use the one fold BCA model variant instead of 5-fold-cross validation."),
    )
    parser.add_argument(
        "--fast-total",
        default=None,
        action="store_true",
        help=("Use TotalSegmentator's fast mode."),
    )
    parser.add_argument(
        "--theme",
        default=None,
        choices=["dark", "light"],
        help=(
            "Theme used for generated BCA PDFs. Defaults to the THEME "
            "environment variable or 'light'."
        ),
    )
    parser.add_argument(
        "-l",
        "--license_number",
        default=None,
        type=str,
        help="TotalSegmentator license number",
    )
    return parser


def run(argv: list[str] | None = None) -> None:
    parser = get_parser()
    args = parser.parse_args(argv)

    logging.basicConfig()
    # Root stays at WARNING so third-party loggers are quiet. body_organ_analysis
    # is pinned to INFO in body_organ_analysis/__init__.py; --verbose controls
    # only the console handler so BOA INFO surfaces in the terminal on demand.
    logging.getLogger().setLevel(logging.WARNING)
    verbose: bool = args.verbose or env_bool("VERBOSE", False)
    console_level = logging.INFO if verbose else logging.WARNING
    for h in logging.getLogger().handlers:
        h.setLevel(console_level)

    # TODO add triton inference logic
    # if args.triton_url is not None:
    #     os.environ["TRITON_URL"] = args.triton_url
    #     os.environ["nnUNet_USE_TRITON"] = "1"
    # else:
    #     os.environ["nnUNet_USE_TRITON"] = "0"

    models_to_compute = args.models
    device = resolve_device(args.device)
    theme: str = args.theme or os.getenv("THEME", "light")
    license_number: str | None = args.license_number or env_str("LICENSE_NUMBER")
    fast_bca: bool = args.fast_bca or env_bool("FAST_BCA", False)
    fast_total: bool = args.fast_total or env_bool("FAST_TOTAL", False)
    bca_no_pdf: bool = args.bca_no_pdf or env_bool("BCA_NO_PDF", False)
    skip_contrast_information: bool = args.skip_contrast_information or env_bool(
        "SKIP_CONTRAST_INFORMATION", False
    )

    if license_number:
        set_license_number(license_number, skip_validation=False)

    # TODO: remove in 1.1.0
    if "PREDICT_FAST" in os.environ:
        warnings.warn(
            "The PREDICT_FAST environment variable is deprecated and will no "
            "longer have any effect starting with version 1.1.0. Use the "
            "FAST_BCA and FAST_TOTAL environment variables (or the --fast-bca "
            "and --fast-total flags) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        fast_bca = True
        fast_total = True

    analyze_ct(
        input_folder=args.input_image,
        processed_output_folder=args.output_dir,
        excel_output_folder=args.output_dir,
        models=models_to_compute,
        compute_contrast_information=not skip_contrast_information,
        total_preview=args.preview,
        nr_thr_resamp=args.nr_thr_resamp,
        nr_thr_saving=args.nr_thr_saving,
        device=device,
        license_number=license_number,
        bca_median_filtering=args.bca_median_filtering,
        bca_examined_body_region=args.bca_examined_body_region,
        bca_pdf=not bca_no_pdf,
        recompute=args.force_recompute,
        nnunet_verbose=args.nnunet_verbose,
        fast_bca=fast_bca,
        fast_total=fast_total,
        cnr_adjustment=args.cnr_adjustment,
        theme=theme,
    )

    if args.radiomics:
        logger.info("Calculating radiomics...")
        st = time.time()
        get_radiomics_features_for_entire_dir(
            args.input_image,
            args.output_dir,
            args.output_dir / "statistics_radiomics.json",
        )
        logger.info("  calculated in %.2fs", time.time() - st)

    if args.use_study_prefix:
        study_name = args.input_image.name.removesuffix(".nii.gz")
        for f in args.output_dir.glob("*"):
            f.rename(f.parent / f"{study_name}_{f.name}")


if __name__ == "__main__":
    run()
