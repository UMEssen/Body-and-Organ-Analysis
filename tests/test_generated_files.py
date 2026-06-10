import unittest
from pathlib import Path

import _state
import pytest
from _paths import OUTPUT_CPU_DIR, OUTPUT_GPU_DIR

# Runs after the heavy GPU/CPU inference in test_cli.py and inspects the files
# BOA wrote to the output directories. Collected near the end via the order hook
# in conftest.py and skipped entirely when that inference did not run, so it is
# a no-op for the fast unit suite (`pytest -m "not integration"`).
pytestmark = pytest.mark.integration


class TestGeneratedFiles(unittest.TestCase):
    """Post-inference checks on the artifacts BOA produced.

    Add assertions inside ``test_gpu_outputs`` / ``test_cpu_outputs`` (or new
    test methods) to validate the generated files. The ``_require_*`` guards
    skip cleanly when the corresponding inference did not run or did not finish.
    """

    def setUp(self) -> None:
        if not _state.has_attempted("TestCLI"):
            self.skipTest("requires TestCLI to have run first")

    def _require_gpu_output(self) -> None:
        if not _state.has_completed("test_dicom_inference_gpu"):
            self.skipTest("requires test_dicom_inference_gpu to succeed")

    def _require_cpu_output(self) -> None:
        if not _state.has_completed("test_nifti_inference_cpu"):
            self.skipTest("requires test_nifti_inference_cpu to succeed")

    def _assert_present(self, path: Path) -> None:
        self.assertTrue(path.is_file(), f"expected generated file is missing: {path}")
        self.assertGreater(path.stat().st_size, 0, f"generated file is empty: {path}")

    def test_gpu_outputs(self) -> None:
        # GPU run command: -m bca+heartchambers_highres --cnr-adjustment
        self._require_gpu_output()
        for name in ("output.xlsx", "debug_information.txt", "total.nii.gz"):
            self._assert_present(OUTPUT_GPU_DIR / name)

        # Further files this run produces, ready to enable as needed:
        #   tissues.nii.gz, body_regions.nii.gz, report.pdf   (from `bca`)
        #   heartchambers.nii.gz                              (heartchambers_highres)
        #   total-measurements.json, bca-measurements.json
        # e.g. self._assert_present(OUTPUT_GPU_DIR / "report.pdf")

    def test_cpu_outputs(self) -> None:
        # CPU run command: -m total --fast-bca --fast-total --bca-no-pdf
        self._require_cpu_output()
        for name in ("output.xlsx", "debug_information.txt", "total.nii.gz"):
            self._assert_present(OUTPUT_CPU_DIR / name)

        # e.g. self._assert_present(OUTPUT_CPU_DIR / "total-measurements.json")


if __name__ == "__main__":
    unittest.main()
