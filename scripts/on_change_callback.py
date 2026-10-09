import json
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import orthanc
from celery_task import analyze_stable_series
from util import get_db_connection, write_to_postgres

IMPORTANT_INFOS = [
    "StudyDate",
    "AccessionNumber",
    "SeriesNumber",
    "SeriesDescription",
]


def summarize_important_info(dicom_tags: dict[str, Any]) -> str:
    info_text = ""
    for info in IMPORTANT_INFOS:
        if info in dicom_tags:
            info_text += f"{info}: {dicom_tags[info]}\n"
        else:
            info_text += f"{info}: Unknown\n"
    return info_text


def split_multi_value(value: Any) -> list[str]:
    """Return the values of a multi-valued tag as a list of strings.

    Orthanc's simplified-tags return multi-valued tags as a single backslash-joined
    string (e.g. "ORIGINAL\\PRIMARY\\AXIAL"), so a plain `in` would be a substring
    check.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split("\\")]
    return [str(part).strip() for part in value]


def is_axial(
    image_orientation: Sequence[float | str] | None, tolerance: float = 0.8
) -> bool:
    """Return True if the slice normal points predominantly along the z-axis.

    Args:
        image_orientation: ImageOrientationPatient (0020,0037), six direction cosines.
        tolerance: Minimum share of the z-component in the slice normal
            (0.8 ~ 37° tilt).
    """
    if image_orientation is None or len(image_orientation) != 6:
        return False
    try:
        rx, ry, rz, cx, cy, cz = (float(value) for value in image_orientation)
    except (TypeError, ValueError):
        return False
    # Cross product of row and column direction = slice normal
    nx, ny, nz = ry * cz - rz * cy, rz * cx - rx * cz, rx * cy - ry * cx
    norm: float = (nx**2 + ny**2 + nz**2) ** 0.5
    return norm > 0 and abs(nz) / norm >= tolerance


def generate_task(
    series_info: dict[str, Any], dicom_tags: dict[str, Any], minimum_images: int = 10
) -> bool:
    if len(series_info["Instances"]) < minimum_images:
        orthanc.LogWarning(
            f"The series has less than {minimum_images} "
            f"instances: {len(series_info['Instances'])}"
        )
        return False

    if "Modality" in dicom_tags and dicom_tags["Modality"] != "CT":
        orthanc.LogWarning(f"The modality is not CT: {dicom_tags['Modality']}")
        return False

    if "ImageType" in dicom_tags and "AXIAL" not in split_multi_value(
        dicom_tags["ImageType"]
    ):
        # Not every vendor writes AXIAL into ImageType
        orientation = dicom_tags.get("ImageOrientationPatient")
        if not is_axial(split_multi_value(orientation)):
            orthanc.LogWarning(
                f"The image type is not 'AXIAL': {dicom_tags['ImageType']} "
                f"and the orientation is not axial: {orientation}"
            )
            return False
        orthanc.LogWarning(
            f"The image type is not 'AXIAL': {dicom_tags['ImageType']}, "
            f"but the orientation is axial: {orientation}"
        )

    return True


def get_max_id(connection: Any) -> Any:
    cursor = connection.cursor()
    cursor.execute("SELECT MAX(id) FROM boa_entries")
    record = cursor.fetchone()
    cursor.close()
    return record[0]


def on_change(change_type: int, _level: int, resource_id: str) -> None:
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
            "orthanc_timestamp": datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
            "study_description": dicom_tags.get("StudyDescription", "Unknown"),
            "accession_number": dicom_tags.get("AccessionNumber", "Unknown"),
            "series_description": dicom_tags.get("SeriesDescription", "Unknown"),
        }
        db_conn = get_db_connection()
        try:
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
                    f"The series {resource_id} was not computed "
                    "because it did not pass the filtering."
                )
                orthanc.RestApiDelete(f"/series/{resource_id}")
        finally:
            if db_conn is not None:
                db_conn.close()


orthanc.RegisterOnChangeCallback(on_change)
