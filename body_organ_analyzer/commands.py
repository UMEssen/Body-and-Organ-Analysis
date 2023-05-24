import logging
import warnings
from pathlib import Path
from time import time
from typing import Dict, List

import pandas as pd
from boa_contrast import predict
from totalsegmentator.util import ADDITIONAL_MODELS_OUTPUT_NAME

from body_organ_analyzer._version import __githash__, __version__
from body_organ_analyzer.compute.bca_metrics import compute_bca_metrics
from body_organ_analyzer.compute.inference import compute_all_models
from body_organ_analyzer.compute.io import get_image_info
from body_organ_analyzer.compute.ts_metrics import compute_segmentator_metrics

logger = logging.getLogger(__name__)
# Suppress PyTorch warnings
warnings.filterwarnings(
    "ignore",
    message="torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling.",
)
warnings.filterwarnings(
    "ignore",
    message="User provided device_type of 'cuda', but CUDA is not available. Disabling",
)


def analyze_ct(
    input_folder: Path,
    processed_output_folder: Path,
    excel_output_folder: Path,
    models: List[str],
    compute_contrast_information: bool = True,
    total_preview: bool = True,
    nr_thr_resamp: int = 1,
    nr_thr_saving: int = 1,
    bca_median_filtering: bool = False,
    bca_examined_body_region: str = None,
    bca_pdf: bool = True,
    bca_compute_bmd: bool = True,
    keep_debug_information: bool = False,
    recompute: bool = False,
) -> Path:
    start_total = time()
    ct_info: List[Dict] = []
    if input_folder.is_file() and ".nii" in input_folder.name:
        ct_path = input_folder
    else:
        ct_path, ct_info = get_image_info(
            input_folder=input_folder,
            output_folder=processed_output_folder,
        )
    ct_info = [
        {"name": "BOAVersion", "value": __version__},
        {"name": "BOAGitHash", "value": __githash__},
    ] + ct_info
    logger.info(f"Image loaded and retrieved: DONE in {time() - start_total:0.5f}s")

    if "bca" in models:
        logger.warning(
            "The body composition analysis weights have not been released yet, "
            "but they are coming soon! The body composition analysis computation will be skipped."
        )
        models = [x for x in models if x != "bca"]

    seg_output = processed_output_folder
    seg_output.mkdir(parents=True, exist_ok=True)

    start = time()
    totalsegmentator_params = dict(
        tta=False,
        preview=total_preview,
        nr_threads_resampling=nr_thr_resamp,
        nr_threads_saving=nr_thr_saving,
        nora_tag="None",
        roi_subset=None,
        quiet=False,
        verbose=False,
        test=0,
        crop_path=None,
    )
    compute_all_models(
        ct_path=ct_path,
        segmentation_folder=seg_output,
        models_to_compute=models,
        force_split_threshold=500,
        totalsegmentator_params=totalsegmentator_params,
        bca_params=dict(
            median_filtering=bca_median_filtering,
            examined_body_region=bca_examined_body_region,
            save_pdf=bca_pdf,
            compute_bmd=bca_compute_bmd,
        ),
        keep_debug_segmentations=keep_debug_information,
        recompute=recompute,
    )
    logger.info(f"All models computed: DONE in {time() - start:0.5f}s")

    aggr_df, slices_df, slices_no_limbs_df, bmd_df = None, None, None, None
    if "bca" in models:
        start = time()
        aggr_df, slices_df, slices_no_limbs_df, bmd_df = compute_bca_metrics(
            output_path=seg_output,
        )
        logger.info(f"Metrics from BCA: DONE in {time() - start:0.5f}s")

    regions_df = None
    if any(a in models for a in list(ADDITIONAL_MODELS_OUTPUT_NAME.keys()) + ["total"]):
        start = time()
        region_information, regions_df = compute_segmentator_metrics(
            ct_path=ct_path,
            segmentation_folder=seg_output,
            store_axes=False,
        )
        logger.info(f"Metrics from TotalSegmentator: DONE in {time() - start:0.5f}s")
        ct_info += region_information

    if compute_contrast_information:
        try:
            start = time()
            contrast_information = predict(
                ct_path=ct_path,
                segmentation_folder=seg_output,
                one_mask_per_file=False,
            )
            logger.info(f"Contrast phase predicted: DONE in {time() - start:0.5f}s")
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
        except AssertionError:
            logger.warning("Contrast phase prediction failed")

    info_df = pd.DataFrame(ct_info).set_index("name")

    excel_path = excel_output_folder / "output.xlsx"

    start = time()
    with pd.ExcelWriter(excel_path) as writer:
        info_df.to_excel(writer, sheet_name="info", header=False)
        if regions_df is not None:
            regions_df.to_excel(writer, sheet_name="regions-statistics", index=False)
        if aggr_df is not None:
            aggr_df.to_excel(
                writer, sheet_name="bca-aggregated_measurements", index=False
            )
        if slices_df is not None:
            slices_df.to_excel(writer, sheet_name="bca-slice_measurements", index=False)
        if slices_no_limbs_df is not None:
            slices_no_limbs_df.to_excel(
                writer, sheet_name="bca-slice_measurements_no_ext", index=False
            )
        if bmd_df is not None:
            bmd_df.to_excel(writer, sheet_name="bmd", index=False)
    logger.info(f"Excel stored: DONE in {time() - start:0.5f}s")
    logger.info(f"Complete CT analysis: DONE in {time() - start_total:0.5f}s")

    return excel_path
