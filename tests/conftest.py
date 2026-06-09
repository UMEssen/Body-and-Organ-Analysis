import io
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pytest
import requests
import SimpleITK as sitk
from _paths import CNR_DICOM_DIR, CNR_NIFTI_FILE, OUTPUT_CPU_DIR, OUTPUT_GPU_DIR

# Make the scripts/ modules (util, imports, celery_task, on_change_callback)
# importable from the unit tests without packaging them.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# A public CT series from The Cancer Imaging Archive (TCIA). It is downloaded
# fresh for the test session and removed again afterwards -- test_images/ is
# throwaway, git-ignored data, so nothing is kept on disk between runs. The
# endpoint (TCIA_API) and series UID (TCIA_UID) match the CI secrets.
_TCIA_GET_IMAGE_URL = os.environ.get(
    "TCIA_API",
    "https://services.cancerimagingarchive.net/nbia-api/services/v1/getImage",
)

_MODULE_ORDER = {"test_cli": 0, "test_results": 2}


def _cleanup_test_data() -> None:
    """Delete everything the session downloads or generates.

    Covers the downloaded DICOMs and converted NIfTI (under ``cnr/``) plus the
    NIfTIs and PDF reports written into the output directories.
    """
    for path in (CNR_DICOM_DIR.parent, OUTPUT_CPU_DIR, OUTPUT_GPU_DIR):
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session")
def ct_test_data(request: pytest.FixtureRequest) -> None:
    """Download the CT DICOM series before the tests run and clean up afterwards.

    The DICOM test consumes the series directory directly; the NIfTI test needs
    a single volume, so the series is also converted to ``image.nii.gz``. The
    finalizer is registered up front so the cleanup also runs if the download
    itself fails.
    """
    request.addfinalizer(_cleanup_test_data)

    # 1) Download + extract the DICOM series into test_images/cnr/dicom.
    CNR_DICOM_DIR.mkdir(parents=True, exist_ok=True)
    resp = requests.get(
        _TCIA_GET_IMAGE_URL,
        params={
            "SeriesInstanceUID": os.environ["TCIA_UID"],
            "NewFileNames": "Yes",
        },
        timeout=120,
    )
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        archive.extractall(CNR_DICOM_DIR)

    # 2) Convert the series to a single NIfTI for the CPU test.
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(CNR_DICOM_DIR)))
    sitk.WriteImage(reader.Execute(), str(CNR_NIFTI_FILE))


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Pin test_cli before everything else and test_results last (cross-module deps)."""
    del config

    def order(item: pytest.Item) -> int:
        mod = getattr(item, "module", None)
        name = mod.__name__.rsplit(".", 1)[-1] if mod is not None else ""
        return _MODULE_ORDER.get(name, 1)

    items.sort(key=order)
