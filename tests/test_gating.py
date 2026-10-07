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


@pytest.mark.parametrize(
    ("instances", "tags", "expected"),
    [
        (5, {"Modality": "CT"}, False),
        (10, {"Modality": "MR"}, False),
        (10, {"Modality": "CT", "ImageType": CORONAL}, False),
        (10, {"Modality": "CT", "ImageType": AXIAL}, True),
        # Modality and ImageType are optional
        (12, {}, True),
    ],
)
def test_generate_task(instances: int, tags: dict[str, Any], expected: bool) -> None:
    series = {"Instances": list(range(instances))}
    assert on_change_callback.generate_task(series, tags) is expected


@pytest.mark.parametrize("line", ["StudyDate: 20240101", "AccessionNumber: Unknown"])
def test_summarize_important_info(line: str) -> None:
    assert line in on_change_callback.summarize_important_info(
        {"StudyDate": "20240101"}
    )
