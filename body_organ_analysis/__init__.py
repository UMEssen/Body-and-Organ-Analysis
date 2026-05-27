import logging
import multiprocessing
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"totalsegmentator(\..*)?"
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"nnunetv2(\..*)?"
)

sys.path.append(str(Path(__file__).resolve().parent / "_external"))

# Configures nnUNet_* env vars on import. MUST come before the
# body_organ_analysis.commands
# import below, which transitively loads totalsegmentator.nnunet -> nnunetv2.paths.
import totalsegmentator.nnunet_env  # noqa
from totalsegmentator.config import set_config_key, setup_totalseg  # noqa: E402

# Fix: RuntimeError: Some background workers are no longer alive
if multiprocessing.current_process().name == "MainProcess":
    setup_totalseg()
    set_config_key("send_usage_stats", False)

from body_organ_analysis._version import __githash__, __version__  # noqa: E402
from body_organ_analysis.commands import analyze_ct  # noqa: E402
from body_organ_analysis.compute.io import store_dicoms, store_excel  # noqa: E402

logging.captureWarnings(True)
logging.getLogger("fontTools").setLevel(logging.WARNING)
# warnings.warn() in library code if the issue is avoidable and the client application
# should be modified to eliminate the warning

# logging.warning() if there is nothing the client application can do about the
# situation, but the event should still be noted

logger = logging.getLogger(__name__)
load_dotenv()

__all__ = [
    "__githash__",
    "__version__",
    "analyze_ct",
    "store_dicoms",
    "store_excel",
]
