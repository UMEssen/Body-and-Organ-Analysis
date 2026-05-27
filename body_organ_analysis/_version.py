from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

try:
    __version__ = _pkg_version("body-organ-analysis")
except PackageNotFoundError:
    __version__ = "N/A"

try:
    from body_organ_analysis._githash import __githash__
except ImportError:
    __githash__ = "N/A"
