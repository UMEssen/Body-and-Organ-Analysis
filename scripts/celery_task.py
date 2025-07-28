import logging
import os
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from time import time
from typing import Any, ClassVar, Dict, List, Optional, Set

import requests
from celery import Celery, bootsteps
from celery.signals import worker_ready, worker_shutdown
from requests.exceptions import HTTPError
from util import (
    build_excel,
    collect_auth,
    download_dicoms_from_orthanc,
    get_db_connection,
    get_dicom_tags,
    get_naming_scheme,
    save_data_persistent,
    write_to_postgres,
)

logger = logging.getLogger(__name__)


HEARTBEAT_FILE = Path("/tmp/worker_heartbeat")
READINESS_FILE = Path("/tmp/worker_ready")


class LivenessProbe(bootsteps.StartStopStep):  # type: ignore[misc]
    requires: ClassVar[Set[str]] = {"celery.worker.components:Timer"}

    def __init__(self, parent: Any, **kwargs: Any) -> None:
        super().__init__(parent, **kwargs)
        self.requests: List[Any] = []
        self.tref: Optional[Any] = None

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


@worker_ready.connect  # type: ignore[misc]
def worker_ready_handler(**_: Any) -> None:
    # logger.debug("Creating readiness file.")
    READINESS_FILE.touch()


@worker_shutdown.connect  # type: ignore[misc]
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
    # Retry at most 5 times
    task_publish_retry_policy={
        "max_retries": 5,
    },
    # If this is not set to 1, the worker somehow has a memory leak
    # https://stackoverflow.com/questions/17541452/celery-does-not-release-memory
    worker_max_tasks_per_child=1,
)
app.steps["worker"].add(LivenessProbe)


@app.task()  # type: ignore[misc]
def analyze_stable_series(resource_id: str) -> Dict[str, Optional[str]]:
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

    initial_dict = {
        "task_id": analyze_stable_series.request.id,
        "start_timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
    }

    try:
        dicom_tags = get_dicom_tags(
            session=session, base_url=base_url, resource_id=resource_id
        )
    except HTTPError:
        traceback.print_exc()
        logger.error("Could not retrieve DICOM tags.")
        dicom_tags = {}

    for key in ["study_description", "accession_number", "series_description"]:
        tag = key.replace("_", " ").title().replace(" ", "")
        if tag in dicom_tags:
            initial_dict[key] = dicom_tags[tag]

    db_conn = get_db_connection()

    if len(dicom_tags) == 0:
        initial_dict["end_timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        initial_dict["computed"] = False
        write_to_postgres(db_conn, initial_dict)

        if db_conn is not None:
            db_conn.close()
        return {"outputs": None}

    write_to_postgres(db_conn, data=initial_dict)

    secondary_excel_path = get_naming_scheme(dicom_tags, patient_info)

    logger.info(f"The target directory is {secondary_excel_path}.")

    start_init = time()
    output_information = ""
    computed = False
    with tempfile.TemporaryDirectory(prefix="boa_") as working_dir:
        if output_root is not None:
            logger.info(
                f"The outputs will be stored in {output_root / secondary_excel_path[1:]}"
            )
            output_folder = output_root / secondary_excel_path[1:]
            output_folder.mkdir(parents=True, exist_ok=True)
        else:
            output_folder = Path(working_dir)
        download_start = time()
        input_data_folder = download_dicoms_from_orthanc(
            session=session,
            output_folder=output_folder,
            base_url=base_url,
            series_intances=dicom_tags["Instances"],
        )
        download_time = time() - download_start
        if len(list(input_data_folder.glob("*.dcm"))) == 0:
            output_information += "No DICOMs could be downloaded for this series.\n\n"
        new_excel_path: Optional[Path] = None
        try:
            fast = False
            if "PREDICT_FAST" in os.environ and os.environ["PREDICT_FAST"].lower() in {
                "true",
                "1",
            }:
                fast = True

            new_excel_path, stats = build_excel(
                input_data_folder=input_data_folder,
                output_folder=output_folder,
                dicom_tags=dicom_tags,
                fast=fast,
            )
            computed = True
        except Exception:
            traceback.print_exc()
            output_information += traceback.format_exc() + "\n\n"
            stats = {}
            logger.error("The Excel build failed.")

        start_store = time()
        save_data_persistent(
            input_data_folder=input_data_folder,
            output_folder=output_folder,
            new_excel_path=new_excel_path,
            secondary_excel_path=secondary_excel_path,
            output_information=output_information,
        )
        stats["download_time"] = download_time
        stats["save_persistent_time"] = time() - start_store
    shutil.rmtree(input_data_folder)
    logger.info(f"Entire pipeline: DONE in {time() - start_init:0.5f}s")

    stats["task_id"] = analyze_stable_series.request.id
    stats["end_timestamp"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    stats["computed"] = computed
    write_to_postgres(db_conn, stats)

    if db_conn is not None:
        db_conn.close()

    # Remove series from orthanc
    delete_response = session.delete(
        f"{base_url}/series/{resource_id}",
    )
    delete_response.raise_for_status()

    return {"outputs": secondary_excel_path[1:]}
