import io
import logging
import os
import re
import shutil
import traceback
from pathlib import Path
from time import time
from typing import Any

import imports
import psycopg2
import requests
from psycopg2 import sql
from unidecode import unidecode

logger = logging.getLogger(__name__)

pydicom, _ = imports.optional_import(module="pydicom")
analyze_ct, _ = imports.optional_import(module="body_organ_analysis", name="analyze_ct")
resolve_models, _ = imports.optional_import(
    module="body_organ_analysis.compute.config", name="resolve_models"
)
resolve_device, _ = imports.optional_import(
    module="body_organ_analysis.compute.config", name="resolve_device"
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
    dicom_tags: dict[str, Any], infos_to_include: list[str]
) -> str:
    layer_info = ""
    for info in infos_to_include:
        if info in dicom_tags:
            layer_info += dicom_tags[info] + "_"
        else:
            layer_info += f"Unknown{info}_"
    # Substitute all characters that might create problems with the filesystem
    return re.sub(r"[^\w\.]", "_", _replace_umlauts(layer_info[:-1]))


def get_naming_scheme(dicom_tags: dict[str, str], patient_info: bool = False) -> str:
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


def collect_auth() -> tuple[str, str]:
    if "ORTHANC_USERNAME" in os.environ and "ORTHANC_PASSWORD" in os.environ:
        return os.environ["ORTHANC_USERNAME"], os.environ["ORTHANC_PASSWORD"]
    elif "ORTHANC__REGISTERED_USERS" in os.environ:
        m = re.search(
            r"\{[\"']([^\"']+)['\"].+[\"']([^\"']+)['\"]\}",
            os.environ["ORTHANC__REGISTERED_USERS"],
        )
        if m is None:
            raise ValueError(
                f"{os.environ['ORTHANC__REGISTERED_USERS']} does not comply "
                "to the regex pattern for authorization."
            )
        return m.group(1), m.group(2)
    else:
        raise ValueError(
            "No authentication information has been provided for the orthanc server."
        )


# TODO contextmanager
def get_db_connection() -> Any | None:
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
            "All environment variables must be defined to connect to the monitoring "
            "database: %s missing.",
            ", ".join(missing_vars),
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
    data: dict[str, Any],
) -> None:
    if db_conn is None:
        return
    if "task_id" not in data:
        raise KeyError("The task_id field must be given to update the row.")
    try:
        keys = list(data.keys())
        values = list(data.values())
        update_keys = [k for k in keys if k != "task_id"]

        query = sql.SQL(
            """
            INSERT INTO boa_entries ({columns})
            VALUES ({placeholders})
            ON CONFLICT (task_id) DO UPDATE
            SET {updates}
            """
        ).format(
            columns=sql.SQL(", ").join(map(sql.Identifier, keys)),
            placeholders=sql.SQL(", ").join(sql.Placeholder() * len(keys)),
            updates=sql.SQL(", ").join(
                sql.SQL("{col} = EXCLUDED.{col}").format(col=sql.Identifier(k))
                for k in update_keys
            ),
        )

        with db_conn.cursor() as cur:
            cur.execute(query, values)
        db_conn.commit()
    except Exception:
        logger.exception("Failed to write monitoring information to postgres.")


def download_dicoms_from_orthanc(
    session: requests.Session,
    output_folder: Path,
    base_url: str,
    series_intances: dict[str, Any],
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
    logger.info("DICOM data store: DONE in %0.5fs", time() - start)
    return input_data_folder


def build_excel(
    input_data_folder: Path,
    output_folder: Path,
    dicom_tags: dict[str, Any],
    fast: bool = False,
) -> tuple[Path, dict[str, Any]]:
    # Setup before calling
    start = time()
    excel_path, stats = analyze_ct(
        input_folder=input_data_folder,
        processed_output_folder=output_folder,
        excel_output_folder=output_folder,
        models=resolve_models(os.environ.get("PACS_MODEL")),
        device=resolve_device(),
        fast=fast,
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
    logger.info("Excel build: DONE in %0.5fs", time() - start)

    return new_excel_path, stats


def save_data_persistent(
    input_data_folder: Path,
    output_folder: Path,
    new_excel_path: Path | None,
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
            logger.exception("Storing segmentation in DicomWeb failed.")
    else:
        logger.info(
            "The variables UPLOAD_USER, UPLOAD_PWD and SEGMENTATION_UPLOAD_URL are "
            "not set, the segmentations will not be uploaded."
        )

    if len(output_information) > 0:
        with (output_folder / "debug_information.txt").open("w") as f:
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
        logger.info("Storing Excel in SMB storage: DONE in %0.5fs", time() - start)
    else:
        logger.info(
            "The variables SMB_USER, SMB_PWD and SMB_DIR_OUTPUT are not set, "
            "the Excel file will not be stored in SMB storage."
        )


def get_dicom_tags(
    session: requests.Session, base_url: str, resource_id: str
) -> dict[str, Any]:
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
