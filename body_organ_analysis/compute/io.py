import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pydicom
import requests
import SimpleITK as sitk

from body_organ_analysis._version import __githash__, __version__
from body_organ_analysis.compute.constants import SERIES_DESCRIPTIONS

logger = logging.getLogger(__name__)


def _get_smb_info() -> Tuple[str, str]:
    smb = os.environ["SMB_DIR_OUTPUT"].replace("\\", "/")
    return smb.split("/")[2], smb[:-1] if smb.endswith("/") else smb


def store_excel(paths_to_store: List[Path], store_path: str) -> None:
    import smbclient.shutil

    smbclient.ClientConfig(
        username=os.environ["SMB_USER"], password=os.environ["SMB_PWD"]
    )
    server_name, full_name = _get_smb_info()
    smbclient.register_session(
        server=server_name,
        username=os.environ["SMB_USER"],
        password=os.environ["SMB_PWD"],
    )
    smbclient.makedirs(f"{full_name + store_path}", exist_ok=True)
    for p in paths_to_store:
        if p.exists():
            smbclient.shutil.copy2(str(p), f"{full_name + store_path + p.name}")
    smbclient.delete_session(server=server_name)


def _set_timestamp(dcm: pydicom.Dataset, timestamp: datetime) -> None:
    dcm.InstanceCreationDate = timestamp.strftime("%Y%m%d")
    dcm.InstanceCreationTime = timestamp.strftime("%H%M%S.%f")
    dcm.SeriesDate = dcm.InstanceCreationDate
    dcm.SeriesTime = dcm.InstanceCreationTime
    dcm.ContentDate = dcm.InstanceCreationDate
    dcm.ContentTime = dcm.InstanceCreationTime


def set_dcm_params(
    img_dcm: pydicom.Dataset,
    out_dcm: pydicom.Dataset,
    series_id: int,
    output_name: str,
    timestamp: datetime,
) -> None:
    out_dcm.SeriesDescription = SERIES_DESCRIPTIONS[output_name]
    # Add enumerate to have different values
    out_dcm.SeriesNumber = 42000 * img_dcm.SeriesNumber + series_id
    out_dcm.SeriesInstanceUID = pydicom.uid.generate_uid(
        entropy_srcs=[
            img_dcm.StudyInstanceUID,  # type: ignore
            img_dcm.SeriesInstanceUID,  # type: ignore
            output_name,
            __githash__,
            __version__,
        ]
    )
    out_dcm.SOPInstanceUID = pydicom.uid.generate_uid(
        entropy_srcs=[
            img_dcm.StudyInstanceUID,
            out_dcm.SeriesInstanceUID,  # type: ignore
        ]
    )
    out_dcm.BodyPartExamined = get_dataset_attr(img_dcm, "BodyPartExamined")
    out_dcm.file_meta.MediaStorageSOPInstanceUID = out_dcm.SOPInstanceUID
    _set_timestamp(out_dcm, timestamp)


def store_dicoms(input_folder: Path, segmentation_folder: Path) -> List[Dict[str, Any]]:
    import pydicom_seg
    from dicomweb_client.api import DICOMwebClient

    generated_dicoms: List[pydicom.Dataset] = []
    image, dicom_files = _load_series_from_disk(input_folder)
    img_dcm = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
    start = time()
    output_dcm_info = []
    timestamp = datetime.now()
    templates = sorted((Path("body_organ_analysis") / "templates").glob("*-meta.json"))
    logger.info("Generating encapsulated PDF...")
    # Write encapsulated PDF DICOM
    if (segmentation_folder / "report.pdf").exists():
        subprocess.call(
            [
                "pdf2dcm",
                "--study-from",
                dicom_files[0],
                "--title",
                "Body Composition Analysis Report",
                str(segmentation_folder / "report.pdf"),
                str(segmentation_folder / "report.dcm"),
            ]
        )
    logger.info("Generating DICOMs...")
    for i, template_name in enumerate(templates):
        output_kind = template_name.name.replace("-meta.json", "")
        seg_file = segmentation_folder / f"{output_kind}.nii.gz"
        if not seg_file.exists():
            logger.warning(f"The segmentation {output_kind} does not exist.")
            continue
        nifti_seg = sitk.ReadImage(str(seg_file))  # type: ignore
        seg_array = sitk.GetArrayFromImage(nifti_seg)  # type: ignore
        assert np.isclose(nifti_seg.GetSize(), image.GetSize()).all(), (  # type: ignore
            f"Image and segmentation {output_kind} do not have the same size: "  # type: ignore
            f"{image.GetSize()} vs. {nifti_seg.GetSize()}"  # type: ignore
        )
        assert np.isclose(nifti_seg.GetSpacing(), image.GetSpacing()).all(), (  # type: ignore
            f"Image and segmentation {output_kind} do not have the same spacing: "  # type: ignore
            f"{image.GetSpacing()} vs. {nifti_seg.GetSpacing()}"  # type: ignore
        )
        if not seg_array.sum():
            logger.warning(f"The segmentation {output_kind} does not have any values.")
            continue
        if output_kind in {"body-regions"}:
            # Remove everything with ignore labels
            seg_array[seg_array == 255] = 0
            nifti_seg_new = sitk.GetImageFromArray(seg_array)
            nifti_seg_new.CopyInformation(nifti_seg)  # type: ignore
            nifti_seg = nifti_seg_new
        # Write DICOM-SEG from ITK image
        with template_name.open("r") as fp:
            template = pydicom_seg.template.from_dcmqi_metainfo(json.load(fp))
        writer = pydicom_seg.MultiClassWriter(
            template=template,
            inplane_cropping=False,
            skip_empty_slices=True,
            skip_missing_segment=False,
        )
        out_dcm = writer.write(nifti_seg, dicom_files)
        set_dcm_params(
            img_dcm=img_dcm,
            out_dcm=out_dcm,
            series_id=i,
            output_name=output_kind,
            timestamp=timestamp,
        )
        output_dcm_info.append(
            {
                "name": output_kind,
                "study_instance_uid": img_dcm.StudyInstanceUID,  # type: ignore
                "series_instance_uid": out_dcm.SeriesInstanceUID,
                "sop_instance_uid": out_dcm.SOPInstanceUID,
            }
        )
        # out_dcm.save_as(segmentation_folder / f"{output_kind}.dcm")
        generated_dicoms.append(out_dcm)
    if (segmentation_folder / "report.dcm").exists():
        pdf_dcm = pydicom.dcmread(segmentation_folder / "report.dcm")
        pdf_dcm.SeriesDescription = "Body Composition Analysis Report"
        set_dcm_params(
            img_dcm=img_dcm,
            out_dcm=pdf_dcm,
            series_id=len(templates),
            output_name="report",
            timestamp=timestamp,
        )
        generated_dicoms.append(pdf_dcm)

    # Use Orthanc DICOM-Web interface to upload files
    new_session = requests.Session()
    new_session.auth = (os.environ["UPLOAD_USER"], os.environ["UPLOAD_PWD"])
    logger.info(
        f"Uploading {len(generated_dicoms)} segmentations to {os.environ['SEGMENTATION_UPLOAD_URL']} "
        f"with user {os.environ['UPLOAD_USER']}."
    )
    client = DICOMwebClient(
        os.environ["SEGMENTATION_UPLOAD_URL"],
        session=new_session,
    )
    client.store_instances(generated_dicoms)
    logger.info(f"Storing results as DICOMS: DONE in {time() - start:0.5f}s")

    return output_dcm_info


def _load_series_from_disk(working_dir: Path) -> Tuple[sitk.Image, List[str]]:
    reader = sitk.ImageSeriesReader()  # type: ignore
    files = reader.GetGDCMSeriesFileNames(str(working_dir))  # type: ignore
    reader.SetFileNames(files)  # type: ignore
    image = reader.Execute()  # type: ignore
    return image, files


def _compute_age(date: datetime, birthdate: datetime) -> int:
    return (
        date.year
        - birthdate.year
        - ((date.month, date.day) < (birthdate.month, birthdate.day))
    )


def get_dataset_attr(dcm: pydicom.Dataset, name: str) -> Any:
    return getattr(dcm, name) if hasattr(dcm, name) else None


def compute_boa(
    dcm: pydicom.Dataset, num_dicoms: int, minimum_images: int = 10
) -> Tuple[bool, str]:
    if (
        get_dataset_attr(dcm, "Modality") is not None
        and get_dataset_attr(dcm, "Modality") != "CT"
    ):
        message = f"The modality is not CT: {get_dataset_attr(dcm, 'Modality')}."
        logger.warning(f"The modality is not CT: {get_dataset_attr(dcm, 'Modality')}.")
        return False, message

    # TODO: Found open source CTs that are SECONDARY, DERIVED, so I have removed the constraints.
    if get_dataset_attr(dcm, "ImageType") is not None and not all(
        typ in get_dataset_attr(dcm, "ImageType")
        for typ in [
            "AXIAL",  # "PRIMARY", "ORIGINAL"
        ]
    ):
        message = (
            f"The image type is not 'AXIAL': " f"{get_dataset_attr(dcm, 'ImageType')}."
        )
        return False, message

    if num_dicoms < minimum_images:
        message = f"The series has less than {minimum_images} instances: {num_dicoms}."
        return False, message

    return True, ""


def get_image_info(
    input_folder: Path, output_folder: Path
) -> Tuple[Path, List[Dict[str, Any]]]:
    # Add code to communicate with PACS and download the image given study/series instance id
    image, dicom_files = _load_series_from_disk(input_folder)
    # Get the DICOM tags
    dcm = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
    compute, message = compute_boa(dcm, len(dicom_files))
    if not compute:
        raise ValueError(message)

    nifti_path = output_folder / "image.nii.gz"
    output_folder.mkdir(parents=True, exist_ok=True)
    # Convert the file to nifti
    sitk.WriteImage(image, str(nifti_path), True)
    ct_info = []
    for name in [
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Date",
        "AgeYears",
        "PatientSex",
        "AccessionNumber",
        "SeriesNumber",
        "SeriesDescription",
        "Modality",
        "CTDIvol",
        "ExposureTime",
        "ExposureTime",
        "XRayTubeCurrent",
        "Exposure",
        "KVP",
        "SpiralPitchFactor",
        "ConvolutionKernel",
        "SliceThickness",
        "PixelSpacing",
        "ScanLength",
    ]:
        # Change the name
        if name == "PatientSex":
            new_name = "Gender"
        else:
            new_name = name
        value: Optional[Any] = None
        # Change the value
        if name == "Date":
            value = (
                datetime.strptime(dcm.SeriesDate, "%Y%m%d").strftime("%d.%m.%Y")
                if get_dataset_attr(dcm, "SeriesDate")
                else None
            )
        elif name == "AgeYears":
            if get_dataset_attr(dcm, "PatientBirthDate") and get_dataset_attr(
                dcm, "SeriesDate"
            ):
                value = _compute_age(
                    date=datetime.strptime(dcm.SeriesDate, "%Y%m%d"),
                    birthdate=datetime.strptime(dcm.PatientBirthDate, "%Y%m%d"),
                )
        elif name == "ConvolutionKernel":
            value = (
                dcm.ConvolutionKernel[0]
                if isinstance(
                    get_dataset_attr(dcm, "ConvolutionKernel"),
                    pydicom.multival.MultiValue,
                )
                else get_dataset_attr(dcm, "ConvolutionKernel")
            )
        elif name == "PixelSpacing":
            if isinstance(
                get_dataset_attr(dcm, "PixelSpacing"), pydicom.multival.MultiValue
            ):
                ct_info.append(
                    {
                        "name": "PixelSpacingX",
                        "value": dcm.PixelSpacing[0],
                    }
                )
                new_name = "PixelSpacingY"
                value = dcm.PixelSpacing[1]
            else:
                value = get_dataset_attr(dcm, "PixelSpacing")
        else:
            value = get_dataset_attr(dcm, name)

        ct_info.append(
            {
                "name": new_name,
                "value": value,
            }
        )
    return nifti_path, ct_info
