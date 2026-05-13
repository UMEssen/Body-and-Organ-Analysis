import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "_external"))

# Configures nnUNet_* env vars on import. MUST come before the body_organ_analysis.commands
# import below, which transitively loads totalsegmentator.nnunet -> nnunetv2.paths.
import totalsegmentator.nnunet_env  # noqa: F401

from body_organ_analysis.commands import analyze_ct
from body_organ_analysis.compute.io import store_dicoms, store_excel

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

__all__ = [
    "analyze_ct",
    "store_dicoms",
    "store_excel",
]
