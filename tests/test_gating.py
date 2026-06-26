import importlib
import os
import sys
import types
import unittest

import pytest

# on_change_callback imports celery (via celery_task) and psycopg2 (via util);
# skip the module if the pacs extra is not installed.
pytest.importorskip("celery")
pytest.importorskip("psycopg2")


def _noop(*_args: object, **_kwargs: object) -> None:
    return None


# on_change_callback does `import orthanc` (provided by the Orthanc runtime) and
# registers a callback at import time. Stub it so the module imports cleanly.
# Attributes are set through a dynamic setattr loop so neither ruff (constant
# setattr) nor mypy (writing an unknown module attribute) complains.
_fake_orthanc = types.ModuleType("orthanc")
_fake_attrs: dict[str, object] = {
    "LogWarning": _noop,
    "LogError": _noop,
    "RegisterOnChangeCallback": _noop,
    "RestApiGet": _noop,
    "RestApiDelete": _noop,
    "ChangeType": types.SimpleNamespace(STABLE_SERIES=1),
}
for _attr, _value in _fake_attrs.items():
    setattr(_fake_orthanc, _attr, _value)
sys.modules.setdefault("orthanc", _fake_orthanc)
os.environ.setdefault("CELERY_BROKER", "memory://")

# scripts/ is placed on sys.path by tests/conftest.py.
on_change_callback = importlib.import_module("on_change_callback")


class TestGenerateTask(unittest.TestCase):
    def test_rejects_too_few_instances(self) -> None:
        series = {"Instances": list(range(5))}
        self.assertFalse(on_change_callback.generate_task(series, {"Modality": "CT"}))

    def test_rejects_non_ct_modality(self) -> None:
        series = {"Instances": list(range(10))}
        self.assertFalse(on_change_callback.generate_task(series, {"Modality": "MR"}))

    def test_rejects_non_axial_image_type(self) -> None:
        series = {"Instances": list(range(10))}
        tags = {"Modality": "CT", "ImageType": ["ORIGINAL", "PRIMARY", "CORONAL"]}
        self.assertFalse(on_change_callback.generate_task(series, tags))

    def test_accepts_valid_axial_ct(self) -> None:
        series = {"Instances": list(range(10))}
        tags = {"Modality": "CT", "ImageType": ["ORIGINAL", "PRIMARY", "AXIAL"]}
        self.assertTrue(on_change_callback.generate_task(series, tags))

    def test_accepts_when_modality_and_image_type_missing(self) -> None:
        series = {"Instances": list(range(12))}
        self.assertTrue(on_change_callback.generate_task(series, {}))


class TestSummarizeInfo(unittest.TestCase):
    def test_known_and_unknown_fields(self) -> None:
        summary = on_change_callback.summarize_important_info({"StudyDate": "20240101"})
        self.assertIn("StudyDate: 20240101", summary)
        self.assertIn("AccessionNumber: Unknown", summary)


if __name__ == "__main__":
    unittest.main()
