"""Method (matcher) registry — classical OpenCV + vismatch passthrough."""

from __future__ import annotations

from ._registry import available, get, register
from . import classical  # noqa: F401  (registers SIFT/RootSIFT/ORB/AKAZE/BRISK)
from . import vismatch_loader  # noqa: F401

__all__ = ["available", "get", "register"]
