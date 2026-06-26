import shutil
from pathlib import Path

# isort: off
import body_organ_analysis  # noqa: F401

from body_composition_analysis.infer.infer import download_bca_weights
from totalsegmentator.config import get_weights_dir
from totalsegmentator.libs import download_pretrained_weights

# isort: on

TOTALSEGMENTATOR_TASKS = (291, 292, 293, 294, 295, 297)
BCA_TASKS = (542, 543)


def strip_macos_junk(weights_dir: Path) -> None:
    """Remove the ``__MACOSX`` dirs and ``._*`` files macOS zips into archives."""
    for macosx_dir in weights_dir.rglob("__MACOSX"):
        if macosx_dir.is_dir():
            shutil.rmtree(macosx_dir, ignore_errors=True)
    for apple_double in weights_dir.rglob("._*"):
        apple_double.unlink(missing_ok=True)
    for apple_double in weights_dir.rglob(".DS_Store"):
        apple_double.unlink(missing_ok=True)


def main() -> None:
    """Download every required TotalSegmentator and BCA dataset, then clean up."""
    for task_id in TOTALSEGMENTATOR_TASKS:
        download_pretrained_weights(task_id)
    for task_id in BCA_TASKS:
        download_bca_weights(task_id)
    strip_macos_junk(get_weights_dir())


if __name__ == "__main__":
    main()
