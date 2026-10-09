import importlib
import os
import sys
from typing import Any
from unittest import mock

import pytest

# on_change_callback imports celery (via celery_task) and psycopg2 (via util);
# skip the module if the pacs extra is not installed.
pytest.importorskip("celery")
pytest.importorskip("psycopg2")

# on_change_callback does `import orthanc` (provided by the Orthanc runtime) and
# registers a callback at import time, so import it against a stub.
with (
    mock.patch.dict(sys.modules, {"orthanc": mock.MagicMock()}),
    mock.patch.dict(os.environ, {"CELERY_BROKER": "memory://"}),
):
    on_change_callback = importlib.import_module("on_change_callback")

AXIAL = ["ORIGINAL", "PRIMARY", "AXIAL"]
CORONAL = ["ORIGINAL", "PRIMARY", "CORONAL"]
NO_PLANE = ["ORIGINAL", "PRIMARY"]

IOP_AXIAL = [1, 0, 0, 0, 1, 0]
IOP_CORONAL = [1, 0, 0, 0, 0, -1]
IOP_SAGITTAL = [0, 1, 0, 0, 0, -1]
IOP_ORTHANC = "1\\0\\0\\0\\1\\0"
# Rotated about the x-axis, i.e. gantry tilt
IOP_TILT_30 = [1, 0, 0, 0, 0.866025, -0.5]
IOP_TILT_45 = [1, 0, 0, 0, 0.707107, -0.707107]


@pytest.mark.parametrize(
    ("instances", "tags", "expected"),
    [
        (5, {"Modality": "CT"}, False),
        (10, {"Modality": "MR"}, False),
        (10, {"Modality": "CT", "ImageType": CORONAL}, False),
        (10, {"Modality": "CT", "ImageType": AXIAL}, True),
        # Orthanc's simplified-tags join multi-valued tags with backslashes
        (10, {"ImageType": "ORIGINAL\\PRIMARY\\AXIAL"}, True),
        (10, {"ImageType": "ORIGINAL\\PRIMARY\\AXIAL "}, True),
        (10, {"ImageType": "ORIGINAL\\PRIMARY\\CORONAL"}, False),
        # Fall back to ImageOrientationPatient if ImageType lacks AXIAL
        (10, {"ImageType": NO_PLANE, "ImageOrientationPatient": IOP_AXIAL}, True),
        (10, {"ImageType": NO_PLANE, "ImageOrientationPatient": IOP_ORTHANC}, True),
        (10, {"ImageType": NO_PLANE, "ImageOrientationPatient": IOP_TILT_30}, True),
        (10, {"ImageType": NO_PLANE, "ImageOrientationPatient": IOP_TILT_45}, False),
        (10, {"ImageType": CORONAL, "ImageOrientationPatient": IOP_CORONAL}, False),
        (10, {"ImageType": NO_PLANE, "ImageOrientationPatient": IOP_SAGITTAL}, False),
        (10, {"ImageType": NO_PLANE}, False),
        # Modality and ImageType are optional
        (12, {}, True),
    ],
)
def test_generate_task(instances: int, tags: dict[str, Any], expected: bool) -> None:
    series = {"Instances": list(range(instances))}
    assert on_change_callback.generate_task(series, tags) is expected


@pytest.mark.parametrize(
    ("orientation", "expected"),
    [
        (IOP_AXIAL, True),
        ([-1, 0, 0, 0, -1, 0], True),
        (["1", "0", "0", "0", "1", "0"], True),
        (IOP_TILT_30, True),
        (IOP_TILT_45, False),
        (IOP_CORONAL, False),
        (IOP_SAGITTAL, False),
        (None, False),
        ([1, 0, 0], False),
        ([1, 0, 0, 1, 0, 0], False),  # Parallel row and column, no normal
        (["1", "0", "0", "0", "1", "abc"], False),
    ],
)
def test_is_axial(orientation: list[float | str] | None, expected: bool) -> None:
    assert on_change_callback.is_axial(orientation) is expected


@pytest.mark.parametrize("line", ["StudyDate: 20240101", "AccessionNumber: Unknown"])
def test_summarize_important_info(line: str) -> None:
    assert line in on_change_callback.summarize_important_info(
        {"StudyDate": "20240101"}
    )
