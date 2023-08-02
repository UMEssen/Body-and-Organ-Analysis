import json

import orthanc
from celery_task import analyze_stable_series

IMPORTANT_INFOS = [
    "StudyDate",
    "AccessionNumber",
    "SeriesNumber",
    "SeriesDescription",
]


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
        info_text = ""
        for info in IMPORTANT_INFOS:
            if info in dicom_tags:
                info_text += f"{info}: {dicom_tags[info]}\n"
            else:
                info_text += f"{info}: Unknown\n"
        orthanc.LogWarning(f"It has the following information:\n{info_text}")
        task_id = analyze_stable_series.delay(
            resource_id=resource_id,
        )
        orthanc.LogWarning(f"The task {task_id} was created for {resource_id}.")


orthanc.RegisterOnChangeCallback(OnChange)
