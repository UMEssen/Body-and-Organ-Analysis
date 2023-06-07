import io
import logging
import os
import re
import shutil
import tempfile
import traceback
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple

import imports
import requests
from celery import Celery, bootsteps
from celery.signals import worker_ready, worker_shutdown
from unidecode import unidecode

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

pydicom, _ = imports.optional_import(module="pydicom")

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

HEARTBEAT_FILE = Path("/tmp/worker_heartbeat")
READINESS_FILE = Path("/tmp/worker_ready")


class LivenessProbe(bootsteps.StartStopStep):
    requires = {"celery.worker.components:Timer"}

    def __init__(self, parent: Any, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self.requests: List = []
        self.tref = None

    def start(self, worker: Any) -> None:
        self.tref = worker.timer.call_repeatedly(
            1.0,
            self.update_heartbeat_file,
            (worker,),
            priority=10,
        )

    def stop(self, worker: Any) -> None:
        # logger.debug("Removing heartbeat file.")
        HEARTBEAT_FILE.unlink(missing_ok=True)

    def update_heartbeat_file(self, _: Any) -> None:
        # logger.debug("Updating heartbeat file.")
        HEARTBEAT_FILE.touch()


@worker_ready.connect
def worker_ready_handler(**_: Any) -> None:
    # logger.debug("Creating readiness file.")
    READINESS_FILE.touch()


@worker_shutdown.connect
def worker_shutdown_handler(**_: Any) -> None:
    # logger.debug("Removing readiness file.")
    READINESS_FILE.unlink(missing_ok=True)


app = Celery(
    broker=os.environ["CELERY_BROKER"],
)

app.conf.update(
    # Make sure that the task is acked upon successful completion
    task_acks_late=True,
    # Only reserve one task per worker process at a time
    worker_prefetch_multiplier=1,
    # The connection pool will be disabled and connections
    # will be established and closed for every use.
    broker_pool_limit=0,
    # Ensure that is always on
    task_publish_retry=True,
    # Retry at most 15 times
    task_publish_retry_policy={
        "max_retries": 15,
    },
)
app.steps["worker"].add(LivenessProbe)


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
                f"{os.environ['ORTHANC__REGISTERED_USERS']} does not comply to the regex patter for authorization."
            )
        return m.group(1), m.group(2)
    else:
        raise ValueError(
            "No authentication information has been provided for the orthanc server."
        )


def download_dicoms_from_orthanc(
    session: requests.Session,
    output_folder: Path,
    base_url: str,
    series_intances: Dict[str, Any],
) -> Path:
    logger.info(f"The outputs will be stored in {output_folder}")
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
) -> Path:
    # Setup before calling
    start = time()
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
    assert isinstance(new_excel_path, Path)
    shutil.move(
        excel_path,
        new_excel_path,
    )
    logger.info(f"Excel build: DONE in {time() - start:0.5f}s")

    return new_excel_path


def save_data_persistent(
    input_data_folder: Path,
    output_folder: Path,
    new_excel_path: Optional[Path],
    secondary_excel_path: str,
) -> None:
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


@app.task(ignore_result=True)
def analyze_stable_series(resource_id: str) -> None:
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
    session.auth = collect_auth()
    base_url = f"{os.environ['ORTHANC_URL']}:{os.environ['ORTHANC_PORT']}"

    # Get all the series information
    series_response = session.get(
        f"{base_url}/series/{resource_id}",
    )

    # TODO: Add checks to filter out the series that are not CTs

    series_response.raise_for_status()
    series_info = series_response.json()

    if not Path("/storage_directory").exists():
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
        output_root = None
    else:
        output_root = Path("/storage_directory")

    # Get info about the patient
    metadata = session.get(
        f"{base_url}/instances/{series_info['Instances'][0]}/metadata?expand"
    ).json()
    dicom_tags = session.get(
        f"{base_url}/instances/{series_info['Instances'][0]}/simplified-tags"
    ).json()
    secondary_excel_path = _get_naming_scheme(metadata, dicom_tags, patient_info)

    start_init = time()
    with tempfile.TemporaryDirectory() as working_dir:
        if output_root is not None:
            logger.info(
                f"The outputs will be stored in {output_root / secondary_excel_path[1:]}"
            )
        else:
            output_root = Path(working_dir)
        input_data_folder = download_dicoms_from_orthanc(
            session=session,
            output_folder=output_root,
            base_url=base_url,
            series_intances=series_info["Instances"],
        )
        try:
            new_excel_path: Optional[Path] = build_excel(
                input_data_folder=input_data_folder,
                output_folder=output_root,
                dicom_tags=dicom_tags,
            )
        except Exception:
            traceback.print_exc()
            new_excel_path = None
            logger.error("The Excel build failed.")

        save_data_persistent(
            input_data_folder=input_data_folder,
            output_folder=output_root,
            new_excel_path=new_excel_path,
            secondary_excel_path=secondary_excel_path,
        )

    logger.info(f"Entire pipeline: DONE in {time() - start_init:0.5f}s")

    # Remove series from orthanc
    delete_response = session.delete(
        f"{base_url}/series/{resource_id}",
    )
    delete_response.raise_for_status()
