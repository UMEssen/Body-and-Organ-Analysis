import logging
import os
import platform
from pathlib import Path

logger = logging.getLogger(__name__)


def configure_weasyprint_runtime() -> None:
    """Make WeasyPrint's native dependencies discoverable on the current OS.

    WeasyPrint relies on Pango (and glib/gobject/harfbuzz/fontconfig) loaded at
    runtime via ``cffi.dlopen``. pip cannot ship those shared libraries, so they
    must be installed out-of-band per OS and made discoverable in-process:

    * Linux   -- no-op; the system loader already searches the standard paths
      (the Docker images install ``libpangocairo-1.0-0``).
    * macOS   -- monkey-patch ``cffi.FFI.dlopen`` to fall back to the Homebrew
      lib dir (dyld does not search it and SIP strips ``DYLD_*``).
    * Windows -- register the GTK3 runtime ``bin`` directory with
      ``os.add_dll_directory`` so the GTK DLLs resolve.

    This must run before ``import weasyprint``.
    """
    system = platform.system()
    if system == "Darwin":
        _configure_macos()
    elif system == "Windows":
        _configure_windows()


def _configure_macos() -> None:
    """Make WeasyPrint's cffi.dlopen calls find Homebrew dylibs on macOS.

    macOS dyld does not search /opt/homebrew/lib by default, and
    DYLD_FALLBACK_LIBRARY_PATH can only be set before the process starts -
    mutating os.environ has no effect on the current interpreter, and on
    Apple Silicon SIP can strip DYLD_* vars from signed binaries.

    Pre-loading the dylibs with ctypes.CDLL(absolute, RTLD_GLOBAL) also does
    not help: macOS does not match a bare leafname in a later dlopen against
    libraries already loaded by absolute path.

    The only reliable in-process fix is to monkey-patch cffi.FFI.dlopen so
    that when WeasyPrint asks for e.g. 'libgobject-2.0.0.dylib' and the
    default search fails, we retry with '/opt/homebrew/lib/libgobject-...'.
    """
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
                return original_dlopen(self, str(candidate), flags)
            raise

    patched_dlopen._boa_patched = True  # type: ignore[attr-defined]
    cffi.FFI.dlopen = patched_dlopen


def _configure_windows() -> None:
    """Register the GTK3 runtime ``bin`` dir so WeasyPrint can load Pango.

    Since Python 3.8 the DLL search path for dependent libraries no longer
    includes ``PATH`` by default, so a GTK3 install that is merely "on PATH" is
    not necessarily found by ``cffi.dlopen``. ``os.add_dll_directory`` is the
    officially supported in-process way to add a search location.

    We look in ``WEASYPRINT_DLL_DIRECTORIES`` (also honoured natively by
    WeasyPrint), the standard GTK3 runtime / MSYS2 install locations, and any
    ``PATH`` entry that actually contains the GTK DLLs, and register the first
    directory that holds them. Registering a single match avoids mixing DLLs
    from two different GTK installs.
    """
    # The GTK leafname WeasyPrint ultimately needs; used to validate candidates.
    marker = "libgobject-2.0-0.dll"

    candidates: list[Path] = []

    # Explicit override (also honoured natively by WeasyPrint).
    env_dirs = os.environ.get("WEASYPRINT_DLL_DIRECTORIES")
    if env_dirs:
        candidates.extend(Path(p) for p in env_dirs.split(os.pathsep) if p)

    # Common GTK3 runtime install locations (Windows env vars are case-insensitive).
    candidates.extend(
        Path(base) / "GTK3-Runtime Win64" / "bin"
        for base in (
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        )
    )

    # MSYS2 toolchains.
    candidates.append(Path(r"C:\msys64\mingw64\bin"))
    candidates.append(Path(r"C:\msys64\ucrt64\bin"))

    # Anything on PATH that actually contains the GTK DLLs.
    candidates.extend(
        Path(entry) for entry in os.environ.get("PATH", "").split(os.pathsep) if entry
    )

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        if not (resolved / marker).exists():
            continue
        os.add_dll_directory(str(resolved))  # type: ignore[attr-defined]  # Windows-only
        logger.debug("Registered GTK3 DLL directory %s for WeasyPrint", resolved)
        return

    logger.warning(
        "GTK3 runtime not found (looked for %s). Install the GTK3 runtime "
        "(https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer)"
        ", set WEASYPRINT_DLL_DIRECTORIES to its `bin` directory, or pass "
        "--bca-no-pdf to skip PDF generation.",
        marker,
    )
