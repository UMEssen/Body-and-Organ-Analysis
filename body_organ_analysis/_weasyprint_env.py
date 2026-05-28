import ctypes.util
import logging
import os
import platform

logger = logging.getLogger(__name__)


def configure_weasyprint_runtime() -> None:
    if platform.system() != "Darwin":
        return
    if ctypes.util.find_library("pango-1.0") is None:
        logger.warning(
            "PDF report generation is enabled but the 'pango' library was not "
            "found. WeasyPrint will fail at runtime. Install it with "
            "`brew install pango` or pass --bca-no-pdf to skip."
        )
        return
    brew_lib = "/opt/homebrew/lib"
    parts = [
        p for p in os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "").split(":") if p
    ]
    if brew_lib not in parts:
        parts.insert(0, brew_lib)
    os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(parts)
