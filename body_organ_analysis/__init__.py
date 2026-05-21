import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent / "_external"))

# Configures nnUNet_* env vars on import. MUST come before the
# body_organ_analysis.commands
# import below, which transitively loads totalsegmentator.nnunet -> nnunetv2.paths.
import totalsegmentator.nnunet_env  # noqa: F401
from totalsegmentator.config import set_config_key, setup_totalseg

setup_totalseg()
set_config_key("send_usage_stats", False)

from body_organ_analysis.commands import analyze_ct  # noqa: E402
from body_organ_analysis.compute.io import store_dicoms, store_excel  # noqa: E402

try:
    from body_organ_analysis._version import __githash__, __version__
except ImportError:
    print('Missing file "body_organ_analysis._version.py"')
    print('Run "./scripts/generate_version.sh" from the project root directory')
    __githash__ = "N/A"
    __version__ = "N/A"

# TODO already in cli.py. Can be removed?
logging.basicConfig()
logging.captureWarnings(True)
# warnings.warn() in library code if the issue is avoidable and the client application
# should be modified to eliminate the warning

# logging.warning() if there is nothing the client application can do about the
# situation, but the event should still be noted

logger = logging.getLogger(__name__)
load_dotenv()

__all__ = [
    "analyze_ct",
    "store_dicoms",
    "store_excel",
]
