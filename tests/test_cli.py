import platform
import shutil
import unittest

import _state
from _paths import CNR_DICOM_DIR, OUTPUT_GPU_DIR

from body_organ_analysis.cli import run


class TestCLI(unittest.TestCase):
    def setUp(self) -> None:
        _state.mark_attempted("TestCLI")

    @unittest.skipUnless(
        platform.system() in {"Windows", "Linux"}, "Requires Windows or Linux"
    )
    def test_dicom_inference_gpu(self) -> None:
        # shutil.rmtree(OUTPUT_GPU_DIR, ignore_errors=True)
        # OUTPUT_GPU_DIR.mkdir()

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
            ]
        )
        _state.mark_complete("test_dicom_inference_gpu")

    def test_nifti_inference_cpu(self) -> None:
        # shutil.rmtree(OUTPUT_CPU_DIR, ignore_errors=True)
        # OUTPUT_CPU_DIR.mkdir()

        # run(
        #     [
        #         "-i",
        #         str(CNR_NIFTI_FILE),
        #         "-o",
        #         str(OUTPUT_CPU_DIR),
        #         "-m",
        #         "bca",
        #         "-d",
        #         "cpu",
        #         "--fast-bca",
        #         "--fast-total",
        #         "--bca-no-pdf",
        #         "--skip-contrast-information",
        #         "--verbose",
        #     ]
        # )
        _state.mark_complete("test_nifti_inference_cpu")

    # Not working yet
    # RuntimeError: ConvTranspose 3D is not supported on MPS
    # def test_dicom_inference_mps(self) -> None: ...


if __name__ == "__main__":
    unittest.main()
