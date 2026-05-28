import ctypes.util
import logging
import platform
import warnings
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from time import time
from typing import Any

import pandas as pd
import SimpleITK as sitk
from boa_contrast import predict
from body_composition_analysis.body_regions.definition import BodyRegion

from body_organ_analysis._version import __githash__, __version__
from body_organ_analysis.compute.bca_metrics import compute_bca_metrics
from body_organ_analysis.compute.inference import compute_all_models
from body_organ_analysis.compute.io import get_image_info
from body_organ_analysis.compute.ts_metrics import compute_segmentator_metrics
from body_organ_analysis.compute.util import ADDITIONAL_MODELS_OUTPUT_NAME

logger = logging.getLogger(__name__)
# Suppress PyTorch warnings
warnings.filterwarnings(
    "ignore",
    message=(
        "torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling."
    ),
)
warnings.filterwarnings(
    "ignore",
    message="User provided device_type of 'cuda', but CUDA is not available. Disabling",
)


@contextmanager
def _debug_log_handler(path: Path, header: str = "") -> Iterator[None]:
    # The header is written straight to the file (bypassing logging) so it
    # appears only in debug_information.txt, never on the console.
    path.write_text(header)
    handler = logging.FileHandler(path, mode="a")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield
    except Exception:
        logger.exception("BOA run failed")
        raise
    finally:
        root.removeHandler(handler)
        handler.close()


def analyze_ct(
    input_folder: Path,
    processed_output_folder: Path,
    excel_output_folder: Path,
    models: Iterable[str],
    compute_contrast_information: bool = True,
    total_preview: bool = True,
    nr_thr_resamp: int = 1,
    nr_thr_saving: int = 6,
    device: str = "gpu",
    bca_median_filtering: bool = False,
    bca_examined_body_region: str | None = None,
    bca_pdf: bool = True,
    recompute: bool = False,
    nnunet_verbose: bool = False,
    fast_bca: bool = False,
    fast_total: bool = False,
    cnr_adjustment: bool = False,
    theme: str = "light",
) -> tuple[Path, dict[str, Any]]:
    processed_output_folder.mkdir(parents=True, exist_ok=True)
    os_name = platform.system()
    header = (
        f"Platform: {os_name}\n"
        f"BOA version: {__version__}\n"
        f"BOA githash: {__githash__}\n"
        f"Device: {device}\n"
        f"Fast BCA: {fast_bca}\n"
        f"Fast Total: {fast_total}\n"
        f"Contrast Prediction: {compute_contrast_information}\n"
        f"PDF generation: {bca_pdf}\n"
        f"Models: {models}\n\n"
    )

    with _debug_log_handler(
        processed_output_folder / "debug_information.txt", header=header
    ):
        if (
            os_name == "Darwin"
            and bca_pdf
            and ctypes.util.find_library("pango-1.0") is None
        ):
            logger.warning(
                "PDF report generation is enabled but the 'pango' library was not "
                "found. WeasyPrint will fail at runtime. Install it with "
                "`brew install pango` or pass --bca-no-pdf to skip."
            )

        start_total = time()
        ct_info: list[dict[str, Any]] = []
        if input_folder.is_file() and ".nii" in input_folder.name.lower():
            ct_path = input_folder
        else:
            ct_path, ct_info = get_image_info(
                input_folder=input_folder,
                output_folder=processed_output_folder,
            )
        ct_info = [
            {"name": "BOAVersion", "value": __version__},
            {"name": "BOAGitHash", "value": __githash__},
            *ct_info,
        ]
        logger.info("Image loaded and retrieved: DONE in %0.5fs", time() - start_total)

        stats: dict[str, Any] = {
            "git_hash": __githash__,
            "boa_version": __version__,
        }
        seg_output = processed_output_folder  # / "segmentations"
        # seg_output.mkdir(parents=True, exist_ok=True)
        start = time()
        totalsegmentator_params = {
            "preview": total_preview,
            "fast": fast_total,
            "ml": True,
            "nr_thr_resamp": nr_thr_resamp,
            "nr_thr_saving": nr_thr_saving,
            "quiet": False,
            "verbose": nnunet_verbose,
            "device": device,
        }
        ct_stats = compute_all_models(
            ct_path=ct_path,
            segmentation_folder=seg_output,
            models_to_compute=models,
            fast_bca=fast_bca,
            force_split_threshold=400,
            totalsegmentator_params=totalsegmentator_params,
            bca_params={
                "median_filtering": bca_median_filtering,
                "examined_body_region": bca_examined_body_region,
                "save_pdf": bca_pdf,
                "theme": theme,
            },
            recompute=recompute,
            cnr_adjustment=cnr_adjustment,
        )
        logger.info("All models computed: DONE in %0.5fs", time() - start)

        stats["inference_time"] = time() - start
        stats.update(ct_stats)

        aggr_df, slices_df, slices_no_limbs_df = None, None, None
        if "bca" in models:
            start = time()
            aggr_df, slices_df, slices_no_limbs_df = compute_bca_metrics(
                output_path=seg_output,
            )
            logger.info("Metrics from BCA: DONE in %0.5fs", time() - start)
            stats["bca_metrics_time"] = time() - start
            regions_path = seg_output / "body_regions.nii.gz"
            if regions_path.is_file():
                regions = sitk.GetArrayFromImage(sitk.ReadImage(regions_path))
                # We store the found regions as a binary integer
                # the first index is the brain
                # the second index is the thorax
                # the third index is the abdomen
                regions_flag = 0
                if BodyRegion.ABDOMINAL_CAVITY in regions:
                    regions_flag = regions_flag | 1
                if BodyRegion.THORACIC_CAVITY in regions:
                    regions_flag = regions_flag | 2
                if BodyRegion.BRAIN in regions:
                    regions_flag = regions_flag | 4
                stats["bca_regions"] = regions_flag

        regions_df = None
        cnr_df = None
        if any(a in models for a in (*ADDITIONAL_MODELS_OUTPUT_NAME, "total")):
            start = time()
            region_information, regions_df, cnr_df = compute_segmentator_metrics(
                ct_path=ct_path,
                segmentation_folder=seg_output,
                store_axes=False,
            )
            logger.info("Metrics from TotalSegmentator: DONE in %0.5fs", time() - start)
            stats["totalsegmentator_metrics_time"] = time() - start
            ct_info += region_information

        if compute_contrast_information:
            try:
                start = time()
                contrast_information = predict(
                    ct_path=ct_path,
                    segmentation_folder=seg_output,
                    one_mask_per_file=False,
                )
                logger.info("Contrast phase predicted: DONE in %0.5fs", time() - start)
                ct_info.append(
                    {
                        "name": "PredictedContrastPhase",
                        "value": contrast_information["phase_ensemble_predicted_class"],
                    }
                )
                ct_info.append(
                    {
                        "name": "PredictedContrastInGIT",
                        "value": contrast_information["git_ensemble_predicted_class"],
                    }
                )
                stats["iv_contrast_phase"] = contrast_information[
                    "phase_ensemble_prediction"
                ]
                stats["git_contrast"] = contrast_information["git_ensemble_prediction"]
            except AssertionError:
                logger.warning("Contrast phase prediction failed")

        info_df = pd.DataFrame(ct_info).set_index("name")

        excel_path = excel_output_folder / "output.xlsx"

        start = time()
        with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
            info_df.to_excel(writer, sheet_name="info", header=False)
            if regions_df is not None:
                regions_df.to_excel(
                    writer, sheet_name="regions-statistics", index=False
                )
            if cnr_df is not None:
                cnr_df.to_excel(
                    writer, sheet_name="cnr-adjusted", startrow=1, index=False
                )
                workbook = writer.book
                worksheet = writer.sheets["cnr-adjusted"]
                warning = (
                    "These results were yielded by a modified version of BOA, "
                    "adjusted for image quality assessment."
                )
                fmt = workbook.add_format(
                    {
                        "bold": True,
                        "bg_color": "#FFF2CC",
                        "align": "center",
                        "text_wrap": True,
                    }
                )
                last_col = len(cnr_df.columns) - 1
                worksheet.merge_range(0, 0, 0, last_col, warning, fmt)
            if aggr_df is not None:
                aggr_df.to_excel(
                    writer, sheet_name="bca-aggregated-measurements", index=False
                )
            if slices_df is not None:
                slices_df.to_excel(
                    writer, sheet_name="bca-slice-measurements", index=False
                )
            if slices_no_limbs_df is not None:
                slices_no_limbs_df.to_excel(
                    writer, sheet_name="bca-slice-measurements_no_ext", index=False
                )
        logger.info("Excel stored: DONE in %0.5fs", time() - start)
        stats["excel_time"] = time() - start
        logger.info("Complete CT analysis: DONE in %0.5fs", time() - start_total)
        stats["total_time"] = time() - start_total

        return excel_path, stats
