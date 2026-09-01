import logging
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from body_organ_analysis.commands import _debug_log_handler
from body_organ_analysis.compute.config import require_license


class TestDebugLogHandler(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "debug_information.txt"
        self.log = logging.getLogger("body_organ_analysis.tests")

    def test_system_exit_is_reported(self) -> None:
        with self.assertRaises(SystemExit), _debug_log_handler(self.path):
            self.log.info("Computing model heartchambers_highres...")
            sys.exit(1)
        contents = self.path.read_text()
        self.assertIn("BOA run exited early with code 1", contents)
        self.assertIn("SystemExit", contents)

    def test_clean_exit_is_not_reported_as_a_failure(self) -> None:
        with self.assertRaises(SystemExit), _debug_log_handler(self.path):
            sys.exit(0)
        self.assertNotIn("exited early", self.path.read_text())

    def test_exception_is_still_reported(self) -> None:
        with self.assertRaises(ValueError), _debug_log_handler(self.path):
            raise ValueError("boom")
        contents = self.path.read_text()
        self.assertIn("BOA run failed", contents)
        self.assertIn("boom", contents)

    def test_keyboard_interrupt_is_reported(self) -> None:
        with self.assertRaises(KeyboardInterrupt), _debug_log_handler(self.path):
            raise KeyboardInterrupt
        self.assertIn("BOA run failed", self.path.read_text())

    def test_write_debug_still_appends_raw_text(self) -> None:
        with _debug_log_handler(self.path) as write_debug:
            write_debug("traceback line")
        self.assertIn("traceback line\n", self.path.read_text())


class TestRequireLicense(unittest.TestCase):
    def test_missing_license_fails_before_inference(self) -> None:
        with (
            mock.patch(
                "totalsegmentator.config.has_valid_license_offline",
                return_value=("missing_license", "ERROR: no license."),
            ),
            self.assertRaises(SystemExit) as ctx,
        ):
            require_license({"total", "heartchambers_highres"})
        self.assertEqual(ctx.exception.code, 1)

    def test_valid_license_passes(self) -> None:
        with mock.patch(
            "totalsegmentator.config.has_valid_license_offline",
            return_value=("yes", "SUCCESS: License is valid."),
        ):
            require_license({"total", "heartchambers_highres"})

    def test_unlicensed_models_skip_the_check(self) -> None:
        with mock.patch("totalsegmentator.config.has_valid_license_offline") as check:
            require_license({"total", "bca"})
        check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
