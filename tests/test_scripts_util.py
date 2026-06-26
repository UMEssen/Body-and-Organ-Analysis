import importlib
import os
import unittest
from unittest import mock

import pytest

# scripts/util.py imports psycopg2 at module load; skip if the pacs extra is not
# installed. scripts/ is placed on sys.path by tests/conftest.py.
pytest.importorskip("psycopg2")
util = importlib.import_module("util")

_TAGS = {
    "CalledAET": "BOA",
    "StudyDate": "20240101",
    "AccessionNumber": "ACC1",
    "StudyDescription": "CT",
    "SeriesNumber": "3",
    "SeriesDescription": "Thorax",
    "PatientName": "Doe",
    "PatientBirthDate": "19700101",
}


class TestReplaceUmlauts(unittest.TestCase):
    def test_german(self) -> None:
        self.assertEqual(util._replace_umlauts("Müller"), "Mueller")
        self.assertEqual(util._replace_umlauts("Straße"), "Strasse")

    def test_scandinavian(self) -> None:
        self.assertEqual(util._replace_umlauts("Grønn"), "Groenn")


class TestProcessInfoElement(unittest.TestCase):
    def test_joins_sanitizes_and_fills_unknowns(self) -> None:
        tags = {"StudyDate": "20240101", "AccessionNumber": "ACC/1"}
        result = util._process_info_element(
            tags, ["StudyDate", "AccessionNumber", "StudyDescription"]
        )
        self.assertEqual(result, "20240101_ACC_1_UnknownStudyDescription")


class TestNamingScheme(unittest.TestCase):
    def test_without_patient_info(self) -> None:
        self.assertEqual(
            util.get_naming_scheme(_TAGS, patient_info=False),
            "/BOA/20240101_ACC1_CT/3_Thorax/",
        )

    def test_with_patient_info(self) -> None:
        self.assertEqual(
            util.get_naming_scheme(_TAGS, patient_info=True),
            "/BOA/Doe_19700101/20240101_ACC1_CT/3_Thorax/",
        )


class TestCollectAuth(unittest.TestCase):
    def test_username_password_env(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"ORTHANC_USERNAME": "u", "ORTHANC_PASSWORD": "p"},
            clear=True,
        ):
            self.assertEqual(util.collect_auth(), ("u", "p"))

    def test_registered_users_fallback(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"ORTHANC__REGISTERED_USERS": '{"alice": "secret"}'},
            clear=True,
        ):
            self.assertEqual(util.collect_auth(), ("alice", "secret"))

    def test_missing_raises(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            util.collect_auth()


if __name__ == "__main__":
    unittest.main()
