import logging
import platform
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_weasyprint_runtime() -> None:
    """Make WeasyPrint's cffi.dlopen calls find Homebrew dylibs on macOS.

    macOS dyld does not search /opt/homebrew/lib by default, and
    DYLD_FALLBACK_LIBRARY_PATH can only be set before the process starts —
    mutating os.environ has no effect on the current interpreter, and on
    Apple Silicon SIP can strip DYLD_* vars from signed binaries.

    Pre-loading the dylibs with ctypes.CDLL(absolute, RTLD_GLOBAL) also does
    not help: macOS does not match a bare leafname in a later dlopen against
    libraries already loaded by absolute path.

    The only reliable in-process fix is to monkey-patch cffi.FFI.dlopen so
    that when WeasyPrint asks for e.g. 'libgobject-2.0.0.dylib' and the
    default search fails, we retry with '/opt/homebrew/lib/libgobject-...'.
    This must run before `import weasyprint`.
    """
    if platform.system() != "Darwin":
        return
    brew_prefix = Path(
        "/opt/homebrew" if platform.machine() == "arm64" else "/usr/local"
    )
    lib_dir = brew_prefix / "lib"
    if not lib_dir.is_dir():
        logger.warning(
            "Homebrew lib dir %s not found. Install pango with `brew install "
            "pango` or pass --bca-no-pdf to skip PDF generation.",
            lib_dir,
        )
        return

    import cffi  # noqa: PLC0415

    if getattr(cffi.FFI.dlopen, "_boa_patched", False):
        return

    original_dlopen = cffi.FFI.dlopen

    def patched_dlopen(self, name: str, flags=0):  # type: ignore[no-untyped-def]
        try:
            return original_dlopen(self, name, flags)
        except OSError:
            # Only retry for bare leafnames, not already-absolute or .dll names.
            if name.startswith("/") or name.endswith(".dll"):
                raise
            candidate = lib_dir / name
            if candidate.exists():
                return original_dlopen(self, candidate, flags)
            raise

    patched_dlopen._boa_patched = True  # type: ignore[attr-defined]
    cffi.FFI.dlopen = patched_dlopen
