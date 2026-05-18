import json
import logging
from pathlib import Path
from typing import Any

import nibabel
import numpy as np
import SimpleITK as sitk
from totalsegmentator.map_to_binary import class_map

from body_composition_analysis.infer.infer import inference
from body_composition_analysis.io import (
    load_image,
    nib_to_sitk,
    process_image,
    sitk_to_nib,
)
from body_composition_analysis.report.builder import AggregatableBodyPart, Builder
from body_composition_analysis.tissue.subclassification import subclassify_tissues

logger = logging.getLogger(__name__)


def create_vertebrae_info(
    total: sitk.Image, detected_body_part: AggregatableBodyPart
) -> dict[str, Any]:
    vertebrae_data = sitk.GetArrayViewFromImage(total)
    vertebrae_map = {
        v.removeprefix("vertebrae_"): k
        for k, v in class_map["total"].items()
        if v.startswith("vertebrae_")
    }
    vertebrae_info = {}
    for vid, label in vertebrae_map.items():
        mask = np.where((vertebrae_data == label).any(axis=(1, 2)))[0]
        if len(mask) == 0:
            continue
        if (
            ("C" in vid and AggregatableBodyPart.NECK not in detected_body_part)
            or ("T" in vid and AggregatableBodyPart.THORAX not in detected_body_part)
            or ("L" in vid and AggregatableBodyPart.ABDOMEN not in detected_body_part)
        ):
            continue
        vertebrae_info[vid] = (int(mask.min()), int(mask.max() + 1))
    return vertebrae_info


# TODO can be removed?
# def compute_segmentation(
#     input_image: Path,
#     output: Path,
#     task_name: str,
#     force_split: bool = False,
#     totalsegmentator_params: dict[str, Any] | None = None,
# ) -> None:
#     totalsegmentator_params = totalsegmentator_params or {}
#     output.parent.mkdir(exist_ok=True, parents=True)
#     inference(
#         ct_path=input_image,
#         output_dir=output,
#         task_name=task_name,
#         recompute=True,
#         force_split=force_split,
#         totalsegmentator_params=totalsegmentator_params,
#     )


def subclassify(
    input_image: Path,
    input_body_regions: Path,
    output: Path,
    median_filtering: bool,
) -> None:
    orientation = nibabel.aff2axcodes(nibabel.load(input_image).affine)
    subclassify_tissues(
        image=sitk.ReadImage(str(input_image)),
        body_regions=sitk.ReadImage(str(input_body_regions)),
        output_dir=output,
        median_filtering=median_filtering,
        orientation=orientation,
    )


def report(
    input_image: Path,
    input_body_parts: Path,
    input_body_regions: Path,
    input_tissues: Path,
    output_report: Path,
    output_measurements: Path,
    examined_body_region: str | None = None,
    save_pdf: bool = True,
) -> None:
    # Load data
    image = load_image(input_image)
    body_parts = load_image(input_body_parts)
    body_regions = load_image(input_body_regions)
    tissues = load_image(input_tissues)
    vertebrae = None

    # Create report
    builder = Builder(image, body_parts, body_regions, tissues)
    if examined_body_region:
        builder.examined_body_part = AggregatableBodyPart[examined_body_region.upper()]
    else:
        builder.examined_body_part = AggregatableBodyPart.from_body_regions(
            body_regions
        )
        if builder.examined_body_part == AggregatableBodyPart.NONE:
            logger.warning("No supported body part detected")

    prepared_data = builder.prepare(vertebrae)
    if save_pdf:
        pdf_bytes = builder.create_pdf("report.html.jinja", **prepared_data)
    json_data = builder.create_json(**prepared_data)

    # Save PDF report
    if save_pdf:
        output_report.parent.mkdir(exist_ok=True, parents=True)
        with output_report.open("wb") as obfile:
            obfile.write(pdf_bytes)
    output_measurements.parent.mkdir(exist_ok=True, parents=True)
    with (output_measurements).open("w") as ofile:
        json.dump(json_data, ofile, indent=2)


def run_pipeline(
    input_image: Path,
    output_dir: Path,
    fast_bca: bool = False,
    examined_body_region: str | None = None,
    median_filtering: bool = False,
    save_pdf: bool = True,
    force_split: bool = False,
    recompute: bool = True,
    crop_body: bool = False,
    totalsegmentator_params: dict[str, Any] | None = None,
) -> None:
    totalsegmentator_params = totalsegmentator_params or {}
    # Write results back to disk
    output_dir.mkdir(exist_ok=True, parents=True)

    # Body Parts Inference
    body_parts = inference(
        ct_path=input_image,
        output_dir=output_dir,
        task_name="body_parts",
        fast_bca=fast_bca,
        recompute=recompute,
        force_split=force_split,
        totalsegmentator_params=totalsegmentator_params,
    )
    # Body Regions Inference
    body_regions = inference(
        ct_path=input_image,
        output_dir=output_dir,
        task_name="body_regions",
        fast_bca=fast_bca,
        recompute=recompute,
        force_split=force_split,
        crop=body_parts if crop_body else None,
        totalsegmentator_params=totalsegmentator_params,
    )
    tissues = subclassify_tissues(
        image=sitk.ReadImage(input_image),
        body_regions=nib_to_sitk(body_regions),
        output_dir=output_dir / "tissues.nii.gz",
        median_filtering=median_filtering,
        orientation=nibabel.aff2axcodes(body_regions.affine),
    )
    # Reload the images with resampling and reordering for visualization
    image = load_image(input_image)
    body_regions = process_image(body_regions)
    body_parts = process_image(body_parts)
    tissues = process_image(sitk_to_nib(tissues))
    total = load_image(output_dir / "total.nii.gz")
    with (output_dir / "total-measurements.json").open("r", encoding="utf-8") as f:
        total_measurements = json.load(f)

    logger.info("All scans have been loaded and preprocessed.")
    # If appropriate perform a vertebrae localization
    if examined_body_region:
        body_part = AggregatableBodyPart[examined_body_region.upper()]
    else:
        body_part = AggregatableBodyPart.from_body_regions(body_regions)
        if body_part == AggregatableBodyPart.NONE:
            logger.warning("No supported body part detected")
        else:
            logger.info(
                "Body parts detected: Abdomen=%s, Thorax=%s, Neck=%s.",
                AggregatableBodyPart.ABDOMEN in body_part,
                AggregatableBodyPart.THORAX in body_part,
                AggregatableBodyPart.NECK in body_part,
            )

    vertebrae_info = create_vertebrae_info(total=total, detected_body_part=body_part)
    # Build report
    builder = Builder(image, body_parts, body_regions, tissues)
    builder.examined_body_part = body_part
    prepared_data = builder.prepare(
        vertebrae_info, total=total, total_measurements=total_measurements
    )
    if save_pdf:
        pdf_bytes = builder.create_pdf("report.html.jinja", **prepared_data)
        with (output_dir / "report.pdf").open("wb") as obfile:
            obfile.write(pdf_bytes)
    json_data = builder.create_json(**prepared_data)

    if vertebrae_info is not None and len(vertebrae_info) > 0:
        with (output_dir / "vertebrae.json").open("w") as ofile:
            json.dump(vertebrae_info, ofile, indent=2)
    with (output_dir / "bca-measurements.json").open("w") as ofile:
        json.dump(json_data, ofile, indent=2)
