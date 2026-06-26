import unittest

import _state
import pandas as pd
import pandas.testing as pdt
import pytest
from _paths import OUTPUT_GPU_DIR, TEST_FOLDER

# Depends on the GPU inference produced by TestCLI (see test_cli.py).
pytestmark = pytest.mark.integration


class TestResults(unittest.TestCase):
    def setUp(self) -> None:
        if not _state.has_attempted("TestCLI"):
            self.skipTest("requires TestCLI to have run first")

    def test_cnr_adjustment(self) -> None:
        if not _state.has_completed("test_dicom_inference_gpu"):
            self.skipTest("requires test_dicom_inference_gpu to succeed")
        gt_path = TEST_FOLDER / "cnr" / "cnr_v1_output.xlsx"
        if not gt_path.is_file():
            self.skipTest(f"ground-truth workbook {gt_path} is not available")
        pred_df = pd.read_excel(
            OUTPUT_GPU_DIR / "output.xlsx", sheet_name="cnr-adjusted", header=1
        )
        gt_df = pd.read_excel(gt_path, sheet_name="cnr-adjusted")
        pdt.assert_frame_equal(pred_df, gt_df, rtol=0.12, atol=0.0)


if __name__ == "__main__":
    unittest.main()
