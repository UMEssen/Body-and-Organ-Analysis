import io
import os
import platform
import zipfile
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest
import requests
import SimpleITK as sitk

from body_organ_analysis.cli import run

# Full inference on a downloaded TCIA series, opt-in via `pytest -m inference`.
# Each output fixture runs BOA once per module; the tests only inspect its files.
pytestmark = [
    pytest.mark.inference,
    pytest.mark.skipif(
        platform.system() not in {"Linux", "Windows"},
        reason="Requires Windows or Linux",
    ),
]

CNR_GROUND_TRUTH = Path(__file__).parents[1] / "test_images/cnr/cnr_v1_output.xlsx"
GENERATED_FILES = ["output.xlsx", "debug_information.txt", "total.nii.gz"]


@pytest.fixture(scope="module")
def dicom_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    resp = requests.get(
        os.environ["TCIA_API"],
        params={"SeriesInstanceUID": os.environ["TCIA_UID"], "NewFileNames": "Yes"},
        timeout=120,
    )
    resp.raise_for_status()
    path = tmp_path_factory.mktemp("dicom")
    with zipfile.ZipFile(io.BytesIO(resp.content)) as archive:
        archive.extractall(path)
    return path


@pytest.fixture(scope="module")
def nifti_file(dicom_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(reader.GetGDCMSeriesFileNames(str(dicom_dir)))
    path = tmp_path_factory.mktemp("nifti") / "image.nii.gz"
    sitk.WriteImage(reader.Execute(), str(path))
    return path


def _run_boa(input_path: Path, output: Path, args: str) -> Path:
    run(["-i", str(input_path), "-o", str(output), "--verbose", *args.split()])
    return output


@pytest.fixture(scope="module")
def gpu_output(dicom_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _run_boa(
        dicom_dir,
        tmp_path_factory.mktemp("output_gpu"),
        "-m bca+heartchambers_highres -d gpu:0 --cnr-adjustment --theme dark",
    )


@pytest.fixture(scope="module")
def cpu_output(nifti_file: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _run_boa(
        nifti_file,
        tmp_path_factory.mktemp("output_cpu"),
        "-m total -d cpu --fast-bca --fast-total --bca-no-pdf "
        "--skip-contrast-information",
    )


@pytest.mark.parametrize("name", GENERATED_FILES)
def test_dicom_inference_gpu(gpu_output: Path, name: str) -> None:
    assert (gpu_output / name).stat().st_size > 0


@pytest.mark.parametrize("name", GENERATED_FILES)
def test_nifti_inference_cpu(cpu_output: Path, name: str) -> None:
    assert (cpu_output / name).stat().st_size > 0


@pytest.mark.skipif(not CNR_GROUND_TRUTH.is_file(), reason="no CNR ground truth")
def test_cnr_adjustment_gpu(gpu_output: Path) -> None:
    pdt.assert_frame_equal(
        pd.read_excel(gpu_output / "output.xlsx", sheet_name="cnr-adjusted", header=1),
        pd.read_excel(CNR_GROUND_TRUTH, sheet_name="cnr-adjusted"),
        rtol=0.12,
        atol=0.0,
    )
