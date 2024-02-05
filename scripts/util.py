import io
import logging
import os
import re
import shutil
import traceback
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple

import imports
import psycopg2
import requests
from unidecode import unidecode

logger = logging.getLogger(__name__)

pydicom, _ = imports.optional_import(module="pydicom")
analyze_ct, _ = imports.optional_import(module="body_organ_analysis", name="analyze_ct")
BASE_MODELS, _ = imports.optional_import(
    module="body_organ_analysis.compute.constants", name="BASE_MODELS"
)

ADDITIONAL_MODELS_OUTPUT_NAME, _ = imports.optional_import(
    module="body_organ_analysis.compute.util", name="ADDITIONAL_MODELS_OUTPUT_NAME"
)

store_dicoms, _ = imports.optional_import(
    module="body_organ_analysis", name="store_dicoms"
)
store_excel, _ = imports.optional_import(
    module="body_organ_analysis", name="store_excel"
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


def get_naming_scheme(dicom_tags: Dict[str, str], patient_info: bool = False) -> str:
    p = f"/{dicom_tags['CalledAET']}"
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


def collect_auth() -> Tuple[str, str]:
    if "ORTHANC_USERNAME" in os.environ and "ORTHANC_PASSWORD" in os.environ:
        return os.environ["ORTHANC_USERNAME"], os.environ["ORTHANC_PASSWORD"]
    elif "ORTHANC__REGISTERED_USERS" in os.environ:
        m = re.search(
            r"\{[\"']([^\"']+)['\"].+[\"']([^\"']+)['\"]\}",
            os.environ["ORTHANC__REGISTERED_USERS"],
        )
        if m is None:
            raise ValueError(
                f"{os.environ['ORTHANC__REGISTERED_USERS']} does not comply to the regex "
                f"pattern for authorization."
            )
        return m.group(1), m.group(2)
    else:
        raise ValueError(
            "No authentication information has been provided for the orthanc server."
        )


def get_db_connection() -> Optional[Any]:
    missing_vars = [
        var
        for var in [
            "POSTGRES_HOST",
            "POSTGRES_PORT",
            "POSTGRES_USER",
            "POSTGRES_PASSWORD",
            "POSTGRES_DATABASE",
        ]
        if var not in os.environ
    ]
    if len(missing_vars) > 0:
        logger.error(
            f"All environment variables must be defined to connect to the monitoring database: {', '.join(missing_vars)} missing."
        )
        return None
    try:
        db_conn = psycopg2.connect(
            host=os.environ["POSTGRES_HOST"],
            port=os.environ["POSTGRES_PORT"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            database=os.environ["POSTGRES_DATABASE"],
        )
        return db_conn
    except Exception:
        logger.error(traceback.format_exc())
        logger.error("Failed to connect to the Postgres database.")
        return None


def write_to_postgres(
    db_conn: Any,
    data: Dict[str, Any],
) -> None:
    if db_conn is None:
        return
    try:
        cur = db_conn.cursor()
        keys, values = zip(*data.items())
        assert "task_id" in keys, "The task_id field must be given to update the row."
        query = f"""
        INSERT INTO boa_entries ({", ".join(keys)})
        VALUES ({", ".join(["'" + str(val) + "'" if isinstance(val, str) else str(val) for val in values])})
        ON CONFLICT (task_id) DO UPDATE
        SET {", ".join(
                [f"{key} = EXCLUDED.{key}" for key in keys if key != "task_id"]
            )
        };
        """
        logger.info(f"Query: {query}")
        cur.execute(query)
        db_conn.commit()
        cur.close()
    except Exception:
        logger.error(traceback.format_exc())
        logger.error("Failed to write monitoring information to postgres.")


def download_dicoms_from_orthanc(
    session: requests.Session,
    output_folder: Path,
    base_url: str,
    series_intances: Dict[str, Any],
) -> Path:
    input_data_folder = output_folder / "input_dicoms"
    input_data_folder.mkdir(parents=True, exist_ok=True)
    start = time()
    for instance_number in series_intances:
        f = session.get(
            f"{base_url}/instances/{instance_number}/file",
        )
        # Parse it using pydicom
        dicom = pydicom.dcmread(io.BytesIO(f.content))
        dicom.save_as(input_data_folder / f"{dicom.SOPInstanceUID}.dcm")
    logger.info(f"DICOM data store: DONE in {time() - start:0.5f}s")
    return input_data_folder


def build_excel(
    input_data_folder: Path, output_folder: Path, dicom_tags: Dict[str, Any]
) -> Tuple[Path, Dict]:
    # Setup before calling
    start = time()
    models = BASE_MODELS + ["bca"]
    excel_path, stats = analyze_ct(
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
    assert isinstance(new_excel_path, Path)
    shutil.move(
        excel_path,
        new_excel_path,
    )
    logger.info(f"Excel build: DONE in {time() - start:0.5f}s")

    return new_excel_path, stats


def save_data_persistent(
    input_data_folder: Path,
    output_folder: Path,
    new_excel_path: Optional[Path],
    secondary_excel_path: str,
    output_information: str,
) -> None:
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
            output_information += traceback.format_exc() + "\n\n"
            logger.error("Storing segmentation in DicomWeb failed.")
    else:
        logger.info(
            "The variables UPLOAD_USER, UPLOAD_PWD and SEGMENTATION_UPLOAD_URL are not set, "
            "the segmentations will not be uploaded."
        )

    if len(output_information) > 0:
        with open(output_folder / "debug_information.txt", "w") as f:
            f.write(output_information)
    if all(
        # Envs need to exist and not be TODO or empty
        env in os.environ and os.environ[env] not in {"", "TODO"}
        for env in ["SMB_USER", "SMB_PWD", "SMB_DIR_OUTPUT"]
    ):
        start = time()
        try:
            if new_excel_path is None:
                store_excel(
                    paths_to_store=[
                        output_folder / "debug_information.txt",
                    ],
                    store_path=secondary_excel_path,
                )
                logger.error("No excel file was generated.")
            else:
                store_excel(
                    paths_to_store=[
                        new_excel_path,
                        output_folder / "report.pdf",
                        output_folder / "preview_total.png",
                        output_folder / "preview_total.pdf",
                        output_folder / "debug_information.txt",
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


def get_dicom_tags(
    session: requests.Session, base_url: str, resource_id: str
) -> Dict[str, Any]:
    # Get all the series information
    series_response = session.get(
        f"{base_url}/series/{resource_id}",
    )
    # TODO: Add checks to filter out the series that are not CTs

    series_response.raise_for_status()
    series_info = series_response.json()
    # Get info about the patient
    metadata = session.get(
        f"{base_url}/instances/{series_info['Instances'][0]}/metadata?expand"
    ).json()
    dicom_tags = session.get(
        f"{base_url}/instances/{series_info['Instances'][0]}/simplified-tags"
    ).json()
    useful_info = {
        "Instances": series_info["Instances"],
        "CalledAET": metadata["CalledAET"],
    }
    for tag in [
        "StudyDate",
        "AccessionNumber",
        "StudyDescription",
        "SeriesNumber",
        "SeriesDescription",
        "PatientName",
        "PatientBirthDate",
    ]:
        if tag in dicom_tags:
            useful_info[tag] = dicom_tags[tag]
    return useful_info
