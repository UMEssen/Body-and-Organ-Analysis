import orthanc
from celery_task import analyze_stable_series

TOKEN = orthanc.GenerateRestApiAuthorizationToken()


def OnChange(change_type: int, level: int, resource_id: str) -> None:
    # Have to wait for this to become a stable series
    if change_type == orthanc.ChangeType.STABLE_SERIES:
        orthanc.LogWarning(f"A new stable series has been received: {resource_id}")
        task_id = analyze_stable_series.delay(
            orthanc_token=TOKEN, resource_id=resource_id
        )
        orthanc.LogWarning(f"The task {task_id} was created for {resource_id}.")


orthanc.RegisterOnChangeCallback(OnChange)
