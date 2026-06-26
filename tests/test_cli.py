import platform
import shutil
import unittest

import _state
import pytest
from _paths import CNR_DICOM_DIR, CNR_NIFTI_FILE, OUTPUT_CPU_DIR, OUTPUT_GPU_DIR

from body_organ_analysis.cli import run

# Full inference on a downloaded TCIA series: needs a GPU, the model weights,
# and network access. Excluded from the fast unit run via `-m "not integration"`.
pytestmark = pytest.mark.integration


@pytest.mark.usefixtures("ct_test_data")
class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        _state.mark_attempted("TestCLI")

    @unittest.skipUnless(
        platform.system() in {"Linux", "Windows"}, "Requires Windows or Linux"
    )
    def test_dicom_inference_gpu(self) -> None:
        shutil.rmtree(OUTPUT_GPU_DIR, ignore_errors=True)
        OUTPUT_GPU_DIR.mkdir()

        run(
            [
                "-i",
                str(CNR_DICOM_DIR),
                "-o",
                str(OUTPUT_GPU_DIR),
                "-m",
                "bca+heartchambers_highres",
                "-d",
                "gpu:0",
                "--cnr-adjustment",
                "--verbose",
                "--theme",
                "dark",
            ]
        )
        _state.mark_complete("test_dicom_inference_gpu")

    @unittest.skipUnless(
        platform.system() in {"Linux", "Windows"}, "Requires Windows or Linux"
    )
    def test_nifti_inference_cpu(self) -> None:
        shutil.rmtree(OUTPUT_CPU_DIR, ignore_errors=True)
        OUTPUT_CPU_DIR.mkdir()

        run(
            [
                "-i",
                str(CNR_NIFTI_FILE),
                "-o",
                str(OUTPUT_CPU_DIR),
                "-m",
                "total",
                "-d",
                "cpu",
                "--fast-bca",
                "--fast-total",
                "--bca-no-pdf",
                "--skip-contrast-information",
                "--verbose",
            ]
        )
        _state.mark_complete("test_nifti_inference_cpu")


if __name__ == "__main__":
    unittest.main()
