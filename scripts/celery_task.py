import io
import logging
import os
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from time import time
from typing import Any, Dict, List

import imports
from celery import Celery
from unidecode import unidecode

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

pydicom, _ = imports.optional_import(module="pydicom")
requests, _ = imports.optional_import(module="requests")

analyze_ct, _ = imports.optional_import(module="body_organ_analyzer", name="analyze_ct")
BASE_MODELS, _ = imports.optional_import(
    module="body_organ_analyzer.compute.constants", name="BASE_MODELS"
)
ADDITIONAL_MODELS_OUTPUT_NAME, _ = imports.optional_import(
    module="totalsegmentator.util", name="ADDITIONAL_MODELS_OUTPUT_NAME"
)

store_dicoms, _ = imports.optional_import(
    module="body_organ_analyzer", name="store_dicoms"
)
store_excel, _ = imports.optional_import(
    module="body_organ_analyzer", name="store_excel"
)

app = Celery(
    broker=os.environ["CELERY_BROKER"],
)


def _replace_umlauts(text: str) -> str:
    vowel_char_map = {
        # German
        ord("ä"): "ae",
        ord("ü"): "ue",
        ord("ö"): "oe",
        ord("ß"): "ss",
        ord("Ä"): "Ae",
        ord("Ü"): "Ue",
        ord("Ö"): "Oe",
        # Scandinavia
        ord("æ"): "ae",
        ord("ø"): "oe",
        ord("å"): "ae",
        ord("Æ"): "Ae",
        ord("Ø"): "Oe",
        ord("Å"): "Ae",
    }
    return unidecode(text.translate(vowel_char_map))  # type: ignore


def _process_info_element(
    dicom_tags: Dict[str, Any], infos_to_include: List[str]
) -> str:
    layer_info = ""
    for info in infos_to_include:
        if info in dicom_tags:
            layer_info += dicom_tags[info] + "_"
        else:
            layer_info += f"Unknown{info}_"
    # Substitute all characters that might create problems with the filesystem
    return re.sub(r"[^\w\.]", "_", _replace_umlauts(layer_info[:-1]))


def _get_naming_scheme(
    metadata: Dict[str, str], dicom_tags: Dict[str, str], patient_info: bool = False
) -> str:
    p = f"/{metadata['CalledAET']}"
    study_layer = _process_info_element(
        dicom_tags, ["StudyDate", "AccessionNumber", "StudyDescription"]
    )
    series_layer = _process_info_element(
        dicom_tags, ["SeriesNumber", "SeriesDescription"]
    )
    if patient_info:
        patient_layer = _process_info_element(
            dicom_tags, ["PatientName", "PatientBirthDate"]
        )
        return f"{p}/{patient_layer}/{study_layer}/{series_layer}/"
    else:
        return f"{p}/{study_layer}/{series_layer}/"


@app.task(ignore_result=True)
def analyze_stable_series(orthanc_token: str, resource_id: str) -> None:
    patient_info = False
    if "PATIENT_INFO_IN_OUTPUT" in os.environ and os.environ[
        "PATIENT_INFO_IN_OUTPUT"
    ].lower() in {
        "true",
        "1",
    }:
        patient_info = True
        logger.warning(
            "CAREFUL: You have selected the PATIENT_INFO_IN_OUTPUT option, this means that the "
            "results will be stored using the name of the patient and the dates of the study. "
            "Also, if your DICOMs are anonymized, "
            "you might not be able to distinguish between different patients."
        )
    # Get the session ready for the next calls
    session = requests.Session()
    session.headers = {"Authorization": orthanc_token}
    base_url = f"{os.environ['ORTHANC_URL']}:{os.environ['ORTHANC_PORT']}"

    # Get all the series information
    series_response = session.get(
        f"{base_url}/series/{resource_id}",
    )
    series_response.raise_for_status()
    series_info = series_response.json()

    output_root = Path("/storage_directory")
    if not output_root.exists():
        if (
            "SMB_DIR_OUTPUT" not in os.environ
            or os.environ["SMB_DIR_OUTPUT"] in {"", "TODO"}
        ) and (
            "SEGMENTATION_UPLOAD_URL" not in os.environ
            or os.environ["SEGMENTATION_UPLOAD_URL"] in {"", "TODO"}
        ):
            raise ValueError(
                "The local directory does not exist and the SMB storage and the "
                "DicomWeb instance have not been specified. "
                "You will not be able to retrieve any of the results."
            )
        if "SMB_DIR_OUTPUT" not in os.environ or os.environ["SMB_DIR_OUTPUT"] in {
            "",
            "TODO",
        }:
            logger.warning(
                "The local directory does not exist and no SMB storage has been specified. "
                "You will not be able to retrieve the Excel results."
            )
        if "SEGMENTATION_UPLOAD_URL" not in os.environ or os.environ[
            "SEGMENTATION_UPLOAD_URL"
        ] in {"", "TODO"}:
            logger.warning(
                "The local directory does not exist and no DicomWeb link has been specified. "
                "You will not be able to retrieve the segmentation results."
            )
        logger.info(
            "Using temporary directory for output instead of given storage "
            "(storage folder does not exist)."
        )
        output_root = Path(tempfile.mktemp())
    metadata = session.get(
        f"{base_url}/instances/{series_info['Instances'][0]}/metadata?expand"
    ).json()
    dicom_tags = session.get(
        f"{base_url}/instances/{series_info['Instances'][0]}/simplified-tags"
    ).json()

    secondary_excel_path = _get_naming_scheme(metadata, dicom_tags, patient_info)
    output_folder = output_root / secondary_excel_path[1:]

    logger.info(f"The outputs will be stored in {output_folder}")
    input_data_folder = output_folder / "input_dicoms"
    input_data_folder.mkdir(parents=True, exist_ok=True)
    start_init = time()
    for instance_number in series_info["Instances"]:
        f = session.get(
            f"{base_url}/instances/{instance_number}/file",
        )
        # Parse it using pydicom
        dicom = pydicom.dcmread(io.BytesIO(f.content))
        dicom.save_as(input_data_folder / f"{dicom.SOPInstanceUID}.dcm")
    logger.info(f"DICOM data store: DONE in {time() - start_init:0.5f}s")

    # Setup before calling
    start = time()
    new_excel_path = None
    try:
        models = BASE_MODELS + list(ADDITIONAL_MODELS_OUTPUT_NAME.keys()) + ["bca"]
        excel_path = analyze_ct(
            input_folder=input_data_folder,
            processed_output_folder=output_folder,
            excel_output_folder=output_folder,
            models=models,
        )
        new_excel_path = excel_path.parent / (
            _process_info_element(
                dicom_tags, ["AccessionNumber", "SeriesNumber", "SeriesDescription"]
            )
            + ".xlsx"
        )
        shutil.move(
            excel_path,
            new_excel_path,
        )
        logger.info(f"Excel build: DONE in {time() - start:0.5f}s")
    except Exception:
        traceback.print_exc()
        logger.error("The Excel build failed.")

    if (
        all(
            # Envs need to exist and not be TODO or empty
            env in os.environ and os.environ[env] not in {"", "TODO"}
            for env in ["SMB_USER", "SMB_PWD", "SMB_DIR_OUTPUT"]
        )
        and new_excel_path is not None
    ):
        start = time()
        try:
            store_excel(
                paths_to_store=[
                    new_excel_path,
                    output_folder / "report.pdf",
                    output_folder / "preview_total.png",
                ],
                store_path=secondary_excel_path,
            )
        except Exception:
            traceback.print_exc()
            logger.error("Storing Excel in SMB storage failed.")
        logger.info(f"Storing Excel in SMB storage: DONE in {time() - start:0.5f}s")
    else:
        logger.info(
            "The variables SMB_USER, SMB_PWD and SMB_DIR_OUTPUT are not set, "
            "the Excel file will not be stored in SMB storage."
        )
    if all(
        # Envs need to exist and not be TODO or empty
        env in os.environ and os.environ[env] not in {"", "TODO"}
        for env in ["UPLOAD_USER", "UPLOAD_PWD", "SEGMENTATION_UPLOAD_URL"]
    ):
        try:
            store_dicoms(
                input_folder=input_data_folder,
                segmentation_folder=output_folder,
            )
        except Exception:
            traceback.print_exc()
            logger.error("Storing segmentation in DicomWeb failed.")
    else:
        logger.info(
            "The variables UPLOAD_USER, UPLOAD_PWD and SEGMENTATION_UPLOAD_URL are not set, "
            "the segmentations will not be uploaded."
        )
    logger.info(f"Entire pipeline: DONE in {time() - start_init:0.5f}s")

    # Remove series from orthanc
    delete_response = session.delete(
        f"{base_url}/series/{resource_id}",
    )
    delete_response.raise_for_status()
    if not output_root.exists():
        shutil.rmtree(output_root)
