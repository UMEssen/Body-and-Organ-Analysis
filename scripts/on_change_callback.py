import json
from datetime import datetime
from typing import Any, Dict

import orthanc
from celery_task import analyze_stable_series
from util import get_db_connection, write_to_postgres

IMPORTANT_INFOS = [
    "StudyDate",
    "AccessionNumber",
    "SeriesNumber",
    "SeriesDescription",
]


def summarize_important_info(dicom_tags: Dict[str, Any]) -> str:
    info_text = ""
    for info in IMPORTANT_INFOS:
        if info in dicom_tags:
            info_text += f"{info}: {dicom_tags[info]}\n"
        else:
            info_text += f"{info}: Unknown\n"
    return info_text


def generate_task(
    series_info: Dict[str, Any], dicom_tags: Dict[str, Any], minimum_images: int = 10
) -> bool:
    if len(series_info["Instances"]) < minimum_images:
        orthanc.LogWarning(
            f"The series has less than {minimum_images} instances: {len(series_info['Instances'])}"
        )
        return False

    if "Modality" in dicom_tags and dicom_tags["Modality"] != "CT":
        orthanc.LogWarning(f"The modality is not CT: {dicom_tags['Modality']}")
        return False

    if "ImageType" in dicom_tags and not all(
        typ in dicom_tags["ImageType"] for typ in ["AXIAL", "PRIMARY", "ORIGINAL"]
    ):
        orthanc.LogWarning(
            f"The image type is not 'ORIGINAL', 'PRIMARY', 'AXIAL': {dicom_tags['ImageType']}"
        )
        return False

    return True


def get_max_id(connection: Any) -> Any:
    cursor = connection.cursor()
    cursor.execute("SELECT MAX(id) FROM boa_entries")
    record = cursor.fetchone()
    cursor.close()
    return record[0]


def OnChange(change_type: int, level: int, resource_id: str) -> None:
    # Have to wait for this to become a stable series
    if change_type == orthanc.ChangeType.STABLE_SERIES:
        orthanc.LogWarning(f"A new stable series has been received: {resource_id}")
        series_info = json.loads(orthanc.RestApiGet(f"/series/{resource_id}"))
        dicom_tags = json.loads(
            orthanc.RestApiGet(
                f"/instances/{series_info['Instances'][0]}/simplified-tags"
            )
        )
        orthanc.LogWarning(
            f"It has the following information:\n{summarize_important_info(dicom_tags)}"
        )

        relevant_infos = {
            "orthanc_timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            "study_description": dicom_tags["StudyDescription"]
            if "StudyDescription" in dicom_tags
            else "Unknown",
            "accession_number": dicom_tags["AccessionNumber"]
            if "AccessionNumber" in dicom_tags
            else "Unknown",
            "series_description": dicom_tags["SeriesDescription"]
            if "SeriesDescription" in dicom_tags
            else "Unknown",
        }
        db_conn = get_db_connection()

        if generate_task(series_info, dicom_tags):
            task_id = analyze_stable_series.delay(
                resource_id=resource_id,
            )
            relevant_infos["task_id"] = str(task_id)
            write_to_postgres(
                db_conn,
                data=relevant_infos,
            )
            orthanc.LogWarning(f"The task {task_id} was created for {resource_id}.")
        else:
            if db_conn is not None:
                relevant_infos["task_id"] = f"none-{get_max_id(db_conn)}"
                relevant_infos["computed"] = False
                write_to_postgres(
                    db_conn,
                    data=relevant_infos,
                )
            orthanc.LogWarning(
                f"The series {resource_id} was not computed because it did not pass the filtering."
            )
            orthanc.RestApiDelete(f"/series/{resource_id}")
        if db_conn is not None:
            db_conn.close()


orthanc.RegisterOnChangeCallback(OnChange)
