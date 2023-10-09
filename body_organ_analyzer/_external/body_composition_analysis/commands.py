import json
import logging
import pathlib
from typing import Any, Dict, Mapping, Optional, Sequence

import nibabel
import numpy as np
import SimpleITK as sitk
from body_composition_analysis.infer.infer import inference
from body_composition_analysis.io import (
    load_image,
    nib_to_sitk,
    process_image,
    sitk_to_nib,
)
from body_composition_analysis.report.builder import AggregatableBodyPart, Builder
from body_composition_analysis.tissue.subclassification import subclassify_tissues
from totalsegmentator.map_to_binary import class_map

logger = logging.getLogger(__name__)


def create_vertebrae_info(
    vertebrae: sitk.Image, detected_body_part: AggregatableBodyPart
):
    vertebrae_data = sitk.GetArrayViewFromImage(vertebrae)
    vertebrae_map = {
        v.replace("vertebrae_", ""): k
        for k, v in class_map["total"].items()
        if "vertebrae" in v
    }
    vertebrae_info = {}
    for vid, label in vertebrae_map.items():
        mask = np.where((vertebrae_data == label).any(axis=(1, 2)))[0]
        if len(mask) == 0:
            continue
        if "C" in vid and AggregatableBodyPart.NECK not in detected_body_part:
            continue
        elif "T" in vid and AggregatableBodyPart.THORAX not in detected_body_part:
            continue
        elif "L" in vid and AggregatableBodyPart.ABDOMEN not in detected_body_part:
            continue
        vertebrae_info[vid] = (int(mask.min()), int(mask.max() + 1))
    return vertebrae_info


def compute_segmentation(
    input_image: pathlib.Path,
    output: pathlib.Path,
    task_name: str,
    force_split: bool = False,
    totalsegmentator_params: Dict[str, Any] = None,
) -> None:
    totalsegmentator_params = totalsegmentator_params or {}
    output.parent.mkdir(exist_ok=True, parents=True)
    inference(
        ct_path=input_image,
        output_dir=output,
        task_name=task_name,
        recompute=True,
        force_split=force_split,
        totalsegmentator_params=totalsegmentator_params,
    )


def subclassify(
    input_image: pathlib.Path,
    input_body_regions: pathlib.Path,
    output: pathlib.Path,
    median_filtering: bool,
):
    orientation = nibabel.aff2axcodes(nibabel.load(input_image).affine)
    subclassify_tissues(
        image=sitk.ReadImage(str(input_image)),
        body_regions=sitk.ReadImage(str(input_body_regions)),
        output_dir=output,
        median_filtering=median_filtering,
        orientation=orientation,
    )


def report(
    input_image: pathlib.Path,
    input_body_parts: pathlib.Path,
    input_body_regions: pathlib.Path,
    input_tissues: pathlib.Path,
    output_report: pathlib.Path,
    output_measurements: pathlib.Path,
    examined_body_region: Optional[str] = None,
    save_pdf: bool = True,
) -> None:
    # Load data
    image = load_image(input_image)
    body_parts = load_image(input_body_parts)
    body_regions = load_image(input_body_regions)
    tissues = load_image(input_tissues)
    vertebrae = None
    bmd = None

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

    prepared_data = builder.prepare(vertebrae, bmd)
    if save_pdf:
        pdf_bytes = builder.create_pdf("report.html.jinja", **prepared_data)
    json_data = builder.create_json(**prepared_data)

    # Save PDF report
    if save_pdf:
        assert output_report.suffix == ".pdf"
        output_report.parent.mkdir(exist_ok=True, parents=True)
        with output_report.open("wb") as obfile:
            obfile.write(pdf_bytes)
    assert output_measurements.suffix == ".json"
    output_measurements.parent.mkdir(exist_ok=True, parents=True)
    with (output_measurements).open("w") as ofile:
        json.dump(json_data, ofile, indent=2)


def run_pipeline(
    input_image: pathlib.Path,
    output_dir: pathlib.Path,
    examined_body_region: Optional[str] = None,
    median_filtering: bool = False,
    compute_bmd: bool = False,
    save_pdf: bool = True,
    force_split: bool = False,
    recompute: bool = True,
    crop_body: bool = False,
    totalsegmentator_params: Dict[str, Any] = None,
) -> None:
    if compute_bmd:
        logger.info("The BMD functionality will be soon part of the BOA, stay tuned!")
    totalsegmentator_params = totalsegmentator_params or {}
    # Write results back to disk
    output_dir.mkdir(exist_ok=True, parents=True)

    # Body Parts Inference
    body_parts = inference(
        ct_path=input_image,
        output_dir=output_dir,
        task_name="body",
        recompute=recompute,
        force_split=force_split,
        postprocess=True,
        totalsegmentator_params=totalsegmentator_params,
    )
    # Perform body region segmentation
    body_regions = inference(
        ct_path=input_image,
        output_dir=output_dir,
        task_name="bca",
        recompute=recompute,
        force_split=force_split,
        crop=body_parts if crop_body else None,
        postprocess=True,
        totalsegmentator_params=totalsegmentator_params,
    )
    # Vertebrae Inference
    vertebrae = inference(
        ct_path=input_image,
        output_dir=output_dir,
        task_name="vertebrae",
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
    vertebrae = process_image(vertebrae)

    total = None
    if (output_dir / "total.nii.gz").exists():
        if not (output_dir / "vertebrae.nii.zg").exists():
            total = vertebrae
        else:
            total = load_image(output_dir / "total.nii.gz")

    total_measurements = None
    if (output_dir / "total-measurements.json").exists():
        with open(output_dir / "total-measurements.json") as ifile:
            total_measurements = json.load(ifile)

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
                f"Body parts detected: "
                f"Abdomen={AggregatableBodyPart.ABDOMEN in body_part}, "
                f"Thorax={AggregatableBodyPart.THORAX in body_part}, "
                f"Neck={AggregatableBodyPart.NECK in body_part}."
            )

    vertebrae_info = create_vertebrae_info(
        vertebrae=vertebrae, detected_body_part=body_part
    )
    bmd = None
    # Build report
    builder = Builder(image, body_parts, body_regions, tissues)
    builder.examined_body_part = body_part
    prepared_data = builder.prepare(
        vertebrae_info, bmd, total=total, total_measurements=total_measurements
    )
    if save_pdf:
        pdf_bytes = builder.create_pdf("report.html.jinja", **prepared_data)
    json_data = builder.create_json(**prepared_data)

    if vertebrae_info is not None and len(vertebrae_info) > 0:
        with (output_dir / "vertebrae.json").open("w") as ofile:
            json.dump(vertebrae_info, ofile, indent=2)
    if bmd is not None:
        with (output_dir / "bmd.json").open("w") as ofile:
            ofile.write(bmd.to_json(indent=2))
    if save_pdf:
        with (output_dir / "report.pdf").open("wb") as obfile:
            obfile.write(pdf_bytes)
    with (output_dir / "bca-measurements.json").open("w") as ofile:
        json.dump(json_data, ofile, indent=2)
