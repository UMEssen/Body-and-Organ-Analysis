import json
import logging
import os
import shutil
from datetime import date, datetime
from pathlib import Path
from time import time
from typing import Any

import numpy as np
import pydicom
import requests
import SimpleITK as sitk
from numpy.typing import NDArray
from pydicom.valuerep import DA, MultiValue

from body_organ_analysis._version import __githash__, __version__
from body_organ_analysis.compute.constants import SERIES_DESCRIPTIONS

logger = logging.getLogger(__name__)


def _get_smb_info() -> tuple[str, str]:
    """
    Parse SMB_DIR_OUTPUT into (server, normalized_path).
    Expects a UNC-style path like \\\\server\\share\\subdir\\ or //server/share/subdir/.
    """
    raw = os.environ["SMB_DIR_OUTPUT"].replace("\\", "/")
    normalized = raw.rstrip("/") + "/"  # ensure single trailing slash
    parts = [p for p in normalized.split("/") if p]
    if len(parts) < 2:
        raise ValueError(
            f"SMB_DIR_OUTPUT must be a UNC-style path with server/share, got: {raw!r}"
        )
    server = parts[0]
    return server, normalized


def _first_if_multi(value: Any) -> Any:
    """Return the first element if value is a MultiValue, else value itself."""
    if isinstance(value, MultiValue):
        return value[0] if len(value) else None
    return value


def _safe_da(value: Any) -> DA | None:
    if not value:
        return None
    try:
        return DA(str(value).strip())
    except (ValueError, TypeError):
        return None


def store_excel(paths_to_store: list[Path], store_path: str) -> None:
    import smbclient  # noqa: PLC0415

    smbclient.ClientConfig(
        username=os.environ["SMB_USER"], password=os.environ["SMB_PWD"]
    )
    server_name, full_name = _get_smb_info()
    smbclient.register_session(
        server=server_name,
        username=os.environ["SMB_USER"],
        password=os.environ["SMB_PWD"],
    )
    try:
        target_dir = f"{full_name}{store_path}"
        smbclient.makedirs(target_dir, exist_ok=True)
        for p in paths_to_store:
            if p.exists():
                smbclient.shutil.copy2(str(p), f"{target_dir}{p.name}")
            else:
                logger.warning("Skipping missing file: %s", p)
    finally:
        smbclient.delete_session(server=server_name)


def _set_timestamp(dcm: pydicom.Dataset, timestamp: datetime) -> None:
    date_str = timestamp.strftime("%Y%m%d")
    time_str = timestamp.strftime("%H%M%S")
    dcm.InstanceCreationDate = date_str
    dcm.InstanceCreationTime = time_str
    dcm.SeriesDate = date_str
    dcm.SeriesTime = time_str
    dcm.ContentDate = date_str
    dcm.ContentTime = time_str


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
            img_dcm.StudyInstanceUID,
            img_dcm.SeriesInstanceUID,
            output_name,
            __githash__,
            __version__,
        ]
    )
    out_dcm.SOPInstanceUID = pydicom.uid.generate_uid(
        entropy_srcs=[
            img_dcm.StudyInstanceUID,
            out_dcm.SeriesInstanceUID,
        ]
    )
    out_dcm.BodyPartExamined = img_dcm.get("BodyPartExamined")
    out_dcm.file_meta.MediaStorageSOPInstanceUID = out_dcm.SOPInstanceUID
    _set_timestamp(out_dcm, timestamp)


def store_dicoms(input_folder: Path, segmentation_folder: Path) -> list[dict[str, Any]]:
    import subprocess  # noqa: PLC0415

    import pydicom_seg  # noqa: PLC0415
    from dicomweb_client.api import DICOMwebClient  # noqa: PLC0415

    generated_dicoms: list[pydicom.Dataset] = []
    image, dicom_files = _load_series_from_disk(input_folder)
    img_dcm = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
    start = time()
    output_dcm_info = []
    timestamp = datetime.now()
    template_dir = Path(__file__).resolve().parent.with_name("templates")
    templates = sorted(template_dir.glob("*-meta.json"))
    if not templates:
        raise RuntimeError(f"No segmentation templates found in {template_dir}")
    logger.info("Generating encapsulated PDF...")
    # Write encapsulated PDF DICOM
    pdf2dcm = shutil.which("pdf2dcm")
    if pdf2dcm is None:
        raise RuntimeError("pdf2dcm not found on PATH")
    if (segmentation_folder / "report.pdf").exists():
        subprocess.run(  # noqa: S603
            [
                pdf2dcm,
                "--study-from",
                dicom_files[0],
                "--title",
                "Body Composition Analysis Report",
                str(segmentation_folder / "report.pdf"),
                str(segmentation_folder / "report.dcm"),
            ],
            check=True,
        )
    logger.info("Generating DICOMs...")
    for i, template_name in enumerate(templates):
        output_kind = template_name.name.replace("-meta.json", "")
        seg_file = segmentation_folder / f"{output_kind}.nii.gz"
        if not seg_file.exists():
            logger.warning("The segmentation %s does not exist.", output_kind)
            continue
        nifti_seg = sitk.ReadImage(seg_file)
        seg_array = sitk.GetArrayFromImage(nifti_seg)
        if not np.isclose(nifti_seg.GetSize(), image.GetSize()).all():
            raise ValueError(
                f"Image and segmentation {output_kind} do not have the same size: "
                f"{image.GetSize()} vs. {nifti_seg.GetSize()}"
            )
        if not np.isclose(nifti_seg.GetSpacing(), image.GetSpacing()).all():
            raise ValueError(
                f"Image and segmentation {output_kind} do not have the same spacing: "
                f"{image.GetSpacing()} vs. {nifti_seg.GetSpacing()}"
            )
        if not seg_array.sum():
            logger.warning("The segmentation %s does not have any values.", output_kind)
            continue
        if output_kind in {"body_regions"}:
            # Remove everything with ignore labels
            seg_array[seg_array == 255] = 0
            nifti_seg_new = sitk.GetImageFromArray(seg_array)
            nifti_seg_new.CopyInformation(nifti_seg)
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
                "study_instance_uid": img_dcm.StudyInstanceUID,
                "series_instance_uid": out_dcm.SeriesInstanceUID,
                "sop_instance_uid": out_dcm.SOPInstanceUID,
            }
        )
        # out_dcm.save_as(segmentation_folder / f"{output_kind}.dcm")
        generated_dicoms.append(out_dcm)
    if (segmentation_folder / "report.dcm").is_file():
        pdf_dcm = pydicom.dcmread(segmentation_folder / "report.dcm")
        pdf_dcm.SeriesDescription = "Body Composition Analysis Report"
        set_dcm_params(
            img_dcm=img_dcm,
            out_dcm=pdf_dcm,
            series_id=len(templates),
            output_name="report",
            timestamp=timestamp,
        )
        output_dcm_info.append(
            {
                "name": "report",
                "study_instance_uid": img_dcm.StudyInstanceUID,
                "series_instance_uid": pdf_dcm.SeriesInstanceUID,
                "sop_instance_uid": pdf_dcm.SOPInstanceUID,
            }
        )
        generated_dicoms.append(pdf_dcm)

    if not generated_dicoms:
        logger.warning("No DICOMs generated. Skipping Orthanc DICOM-Web upload.")
        return output_dcm_info

    # Use Orthanc DICOM-Web interface to upload files
    new_session = requests.Session()
    new_session.auth = (os.environ["UPLOAD_USER"], os.environ["UPLOAD_PWD"])
    logger.info(
        "Uploading %s segmentations to %s with user %s.",
        len(generated_dicoms),
        os.environ["SEGMENTATION_UPLOAD_URL"],
        os.environ["UPLOAD_USER"],
    )
    client = DICOMwebClient(
        os.environ["SEGMENTATION_UPLOAD_URL"],
        session=new_session,
    )
    client.store_instances(generated_dicoms)
    logger.info("Storing results as DICOMS: DONE in %0.5fs", time() - start)

    return output_dcm_info


def _load_series_from_disk(working_dir: Path) -> tuple[sitk.Image, list[str]]:
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(working_dir))
    reader.SetFileNames(files)
    image = reader.Execute()
    return image, files


def _compute_age(date: date, birthdate: date) -> int:
    return (
        date.year
        - birthdate.year
        - ((date.month, date.day) < (birthdate.month, birthdate.day))
    )


def classify_orientation(
    iop: NDArray[np.float64] | None,
) -> tuple[str | None, NDArray[np.float_] | None]:
    if iop is None or len(iop) != 6:
        return None, None
    row = np.asarray(iop[:3], dtype=float)
    col = np.asarray(iop[3:], dtype=float)
    normal = np.cross(row, col)
    ax, ay, az = abs(normal[0]), abs(normal[1]), abs(normal[2])
    if az >= ax and az >= ay:
        return "axial", normal
    if ay >= ax and ay >= az:
        return "coronal", normal
    return "sagittal", normal


def validate_dicom(
    dcm: pydicom.Dataset,
    num_dicoms: int,
    *,
    minimum_images: int = 10,
    axial_normal_z_min: float = 0.85,
) -> str | None:
    # How close to "pure" axial we accept. 1.0 = perfectly aligned with Z.
    # cos(15°) ≈ 0.966; cos(30°) ≈ 0.866. Pick based on how much tilt you tolerate.
    if num_dicoms < minimum_images:
        return f"The series has less than {minimum_images} instances: {num_dicoms}."

    modality = dcm.get("Modality")
    if modality is not None and modality != "CT":
        return f"The modality is not CT: {modality}"

    iop = dcm.get("ImageOrientationPatient")
    if iop is not None:
        plane, normal = classify_orientation(iop)
        if plane is not None and normal is not None and plane != "axial":
            return (
                f"Image plane is {plane}, not axial. "
                f"IOP={list(iop)}, slice normal={normal.tolist()}"
            )
        if normal is not None and abs(normal[2]) < axial_normal_z_min:
            return (
                "Axial but tilted beyond tolerance: |normal_z|="
                f"{abs(normal[2]):.3f} < {axial_normal_z_min}. IOP={list(iop)}"
            )

    # Secondary sanity check on ImageType — only used to *reject* reformats,
    # not to require "AXIAL" (many genuine acquisitions don't include it).
    image_type = set(dcm.get("ImageType") or ())
    bad_markers = {"LOCALIZER", "REFORMATTED", "DERIVED", "PROJECTION IMAGE"}
    hits = bad_markers & image_type
    if hits:
        return f"ImageType contains disqualifying marker(s) {hits}: {list(image_type)}"
    return None


def get_image_info(
    input_folder: Path, output_folder: Path
) -> tuple[Path, list[dict[str, Any]]]:
    image, dicom_files = _load_series_from_disk(input_folder)
    dcm = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)

    message = validate_dicom(dcm, len(dicom_files))
    if message:
        raise ValueError(message)

    output_folder.mkdir(parents=True, exist_ok=True)
    nifti_path = output_folder / "image.nii.gz"
    sitk.WriteImage(image, str(nifti_path), useCompression=True)

    series_date = _safe_da(dcm.get("SeriesDate"))
    birth_date = _safe_da(dcm.get("PatientBirthDate"))
    pixel_spacing = dcm.get("PixelSpacing")

    # (output_name, value) pairs, in display order
    ordered: list[tuple[str, Any]] = [
        ("StudyInstanceUID", dcm.get("StudyInstanceUID")),
        ("SeriesInstanceUID", dcm.get("SeriesInstanceUID")),
        ("Date", series_date.strftime("%d.%m.%Y") if series_date else None),
        (
            "AgeYears",
            _compute_age(date=series_date, birthdate=birth_date)
            if series_date and birth_date
            else None,
        ),
        ("Gender", dcm.get("PatientSex")),
        ("AccessionNumber", dcm.get("AccessionNumber")),
        ("SeriesNumber", dcm.get("SeriesNumber")),
        ("SeriesDescription", dcm.get("SeriesDescription")),
        ("Modality", dcm.get("Modality")),
        ("CTDIvol", dcm.get("CTDIvol")),
        ("ExposureTime", dcm.get("ExposureTime")),
        ("XRayTubeCurrent", dcm.get("XRayTubeCurrent")),
        ("Exposure", dcm.get("Exposure")),
        ("KVP", dcm.get("KVP")),
        ("SpiralPitchFactor", dcm.get("SpiralPitchFactor")),
        (
            "ConvolutionKernel",
            _first_if_multi(dcm.get("ConvolutionKernel")),
        ),
        ("SliceThickness", dcm.get("SliceThickness")),
    ]

    # PixelSpacing -> split into X / Y if multi-valued
    if isinstance(pixel_spacing, MultiValue) and len(pixel_spacing) >= 2:
        ordered.append(("PixelSpacingX", pixel_spacing[0]))
        ordered.append(("PixelSpacingY", pixel_spacing[1]))
    else:
        ordered.append(("PixelSpacing", pixel_spacing))

    ordered.append(("ScanLength", dcm.get("ScanLength")))

    ct_info = [{"name": name, "value": value} for name, value in ordered]
    return nifti_path, ct_info
