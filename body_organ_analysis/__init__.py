import logging
import multiprocessing
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

logging.captureWarnings(True)
# Always emit INFO from BOA loggers; the sinks (console/file) decide visibility.
# body_composition_analysis is BOA's own vendored code (imported top-level via
# the _external sys.path shim), so pin it too -- otherwise its INFO inherits the
# WARNING root set in cli.py and is dropped.
logging.getLogger("body_organ_analysis").setLevel(logging.INFO)
logging.getLogger("body_composition_analysis").setLevel(logging.INFO)
# Suppress fontTools logging from weasyprint
logging.getLogger("fontTools").propagate = False

warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"totalsegmentator(\..*)?"
)
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module=r"nnunetv2(\..*)?"
)

sys.path.append(str(Path(__file__).resolve().parent / "_external"))

load_dotenv()

# Configures nnUNet_* env vars on import. MUST come before the
# body_organ_analysis.commands
# import below, which transitively loads totalsegmentator.nnunet -> nnunetv2.paths.
from totalsegmentator.config import (  # noqa: E402
    set_config_key,
    setup_nnunet,
    setup_totalseg,
)

# Fix: RuntimeError: Some background workers are no longer alive
if multiprocessing.current_process().name == "MainProcess":
    setup_nnunet()
    setup_totalseg()
    set_config_key("send_usage_stats", False)

from body_organ_analysis._version import __githash__, __version__  # noqa: E402
from body_organ_analysis.commands import analyze_ct  # noqa: E402
from body_organ_analysis.compute.io import store_dicoms, store_excel  # noqa: E402

__all__ = [
    "__githash__",
    "__version__",
    "analyze_ct",
    "store_dicoms",
    "store_excel",
]
