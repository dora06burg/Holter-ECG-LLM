from __future__ import annotations

"""
Utilities for working with Lab Streaming Layer (LSL) in this repo.

Why this exists
---------------
On some Linux environments, `pip install pylsl` installs the Python bindings
but *not* the `liblsl` runtime shared library. Also, `liblsl` itself may have
runtime deps (e.g., `libpugixml`).

This project vendors prebuilt runtime `.so` files under:
  ECG_Model/code_for_stream/vendor/lsl_lib/

We load the dependency library via `ctypes` (RTLD_GLOBAL) and point pylsl to
`liblsl` via the `PYLSL_LIB` env var. This avoids requiring users to modify
`LD_LIBRARY_PATH` before launching Python.
"""

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class VendoredLSLPaths:
    lib_dir: Path
    liblsl: Path
    libpugixml: Optional[Path]


def _default_vendored_paths() -> VendoredLSLPaths:
    root = Path(__file__).resolve().parent
    lib_dir = root / "vendor" / "lsl_lib"
    liblsl = lib_dir / "liblsl.so.1.16.2"
    libpugixml = lib_dir / "libpugixml.so.1.10"
    if not libpugixml.exists():
        libpugixml = None
    return VendoredLSLPaths(lib_dir=lib_dir, liblsl=liblsl, libpugixml=libpugixml)


def _try_import_pylsl():
    import pylsl  # type: ignore

    return pylsl


def import_pylsl(*, prefer_vendored: bool = True):
    """
    Import `pylsl`, attempting to fall back to vendored `liblsl` binaries.

    Parameters
    ----------
    prefer_vendored:
        If True, try normal `import pylsl` first; on failure, load vendored
        libraries and retry.
    """
    if not prefer_vendored:
        return _try_import_pylsl()

    try:
        return _try_import_pylsl()
    except Exception:
        paths = _default_vendored_paths()
        if not paths.liblsl.exists():
            raise

        # Load dependency libs into the global symbol table first.
        if paths.libpugixml is not None and paths.libpugixml.exists():
            ctypes.CDLL(str(paths.libpugixml), mode=ctypes.RTLD_GLOBAL)

        os.environ.setdefault("PYLSL_LIB", str(paths.liblsl))
        return _try_import_pylsl()

