import unittest

import _state
import pandas as pd
import pandas.testing as pdt
from _paths import OUTPUT_GPU_DIR, TEST_FOLDER


class TestResults(unittest.TestCase):
    def setUp(self) -> None:
        if not _state.has_attempted("TestCLI"):
            self.skipTest("requires TestCLI to have run first")

    def test_cnr_adjustment(self) -> None:
        if not _state.has_completed("test_dicom_inference_gpu"):
            self.skipTest("requires test_dicom_inference_gpu to succeed")
        pred_df = pd.read_excel(
            OUTPUT_GPU_DIR / "output.xlsx", sheet_name="cnr-adjusted", header=1
        )
        gt_df = pd.read_excel(
            TEST_FOLDER / "cnr" / "cnr_v1_output.xlsx", sheet_name="cnr-adjusted"
        )
        pdt.assert_frame_equal(pred_df, gt_df, rtol=0.12, atol=0.0)


if __name__ == "__main__":
    unittest.main()
