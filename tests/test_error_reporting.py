import sys
from pathlib import Path
from unittest import mock

import pytest

from body_organ_analysis.commands import _debug_log_handler
from body_organ_analysis.compute.config import require_license

LICENSE_CHECK = "totalsegmentator.config.has_valid_license_offline"


@pytest.fixture
def debug_file(tmp_path: Path) -> Path:
    return tmp_path / "debug_information.txt"


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (SystemExit(1), "BOA run exited early with code 1"),
        (SystemExit(1), "SystemExit"),
        (ValueError("boom"), "BOA run failed"),
        (ValueError("boom"), "boom"),
        (KeyboardInterrupt(), "BOA run failed"),
    ],
)
def test_abnormal_exit_is_reported(
    debug_file: Path, error: BaseException, expected: str
) -> None:
    with pytest.raises(type(error)), _debug_log_handler(debug_file):
        raise error
    assert expected in debug_file.read_text()


def test_clean_exit_is_not_reported_as_a_failure(debug_file: Path) -> None:
    with pytest.raises(SystemExit), _debug_log_handler(debug_file):
        sys.exit(0)
    assert "exited early" not in debug_file.read_text()


def test_write_debug_still_appends_raw_text(debug_file: Path) -> None:
    with _debug_log_handler(debug_file) as write_debug:
        write_debug("traceback line")
    assert "traceback line\n" in debug_file.read_text()


def test_missing_license_fails_before_inference() -> None:
    with (
        mock.patch(LICENSE_CHECK, return_value=("missing_license", "ERROR")),
        pytest.raises(SystemExit, match="1"),
    ):
        require_license({"total", "heartchambers_highres"})


def test_valid_license_passes() -> None:
    with mock.patch(LICENSE_CHECK, return_value=("yes", "SUCCESS")):
        require_license({"total", "heartchambers_highres"})


def test_unlicensed_models_skip_the_check() -> None:
    with mock.patch(LICENSE_CHECK) as check:
        require_license({"total", "bca"})
    check.assert_not_called()
