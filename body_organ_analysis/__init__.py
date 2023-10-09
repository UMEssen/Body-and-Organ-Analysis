import logging
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "_external"))

from body_organ_analysis.commands import analyze_ct  # noqa: E402
from body_organ_analysis.compute.io import store_dicoms, store_excel  # noqa: E402

try:
    from body_organ_analysis._version import __githash__, __version__
except ImportError:
    print('Missing file "body_organ_analysis._version.py"')
    print('Run "./scripts/generate_version.sh" from the project root directory')
    __githash__ = "N/A"
    __version__ = "N/A"

logging.basicConfig()
logging.captureWarnings(True)
# warnings.warn() in library code if the issue is avoidable and the client application
# should be modified to eliminate the warning

# logging.warning() if there is nothing the client application can do about the situation,
# but the event should still be noted

logger = logging.getLogger(__name__)


__all__ = [
    "analyze_ct",
    "store_excel",
    "store_dicoms",
]
