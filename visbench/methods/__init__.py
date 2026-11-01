"""Method (matcher) registry — classical OpenCV, vismatch passthrough, romaxx adapter.

The classical methods build on vismatch's BaseMatcher. When vismatch is not
installed (for example a RoMaXX-only environment running `romaxx:` methods),
they are skipped and the reason is kept in IMPORT_ERRORS instead of failing
the whole package import.
"""

from __future__ import annotations

from ._registry import available, get, register
from . import vismatch_loader  # noqa: F401

IMPORT_ERRORS: dict[str, str] = {}

try:
    from . import classical  # noqa: F401  (registers SIFT/RootSIFT/ORB/AKAZE/BRISK)
except ImportError as _e:
    IMPORT_ERRORS["classical"] = f"{type(_e).__name__}: {_e}"

__all__ = ["available", "get", "register", "IMPORT_ERRORS"]
