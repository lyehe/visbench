"""Dataset plugin registry — each module self-registers at import time.

Module-import errors are collected into `IMPORT_ERRORS` rather than raising
during package import; otherwise a single broken plugin would prevent the
entire registry from loading. Errors are also warned-once so they surface in
CI/tests. Set VISBENCH_STRICT_IMPORT=1 to make any plugin failure fatal.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import warnings
from pathlib import Path

from ._registry import DatasetSpec, available, get, register  # noqa: F401

IMPORT_ERRORS: dict[str, str] = {}

_pkg_dir = Path(__file__).parent
_strict = os.environ.get("VISBENCH_STRICT_IMPORT") == "1"
for _m in pkgutil.iter_modules([str(_pkg_dir)]):
    if _m.name.startswith("_"):
        continue
    try:
        importlib.import_module(f"{__name__}.{_m.name}")
    except Exception as _e:  # noqa: BLE001
        IMPORT_ERRORS[_m.name] = f"{type(_e).__name__}: {_e}"
        if _strict:
            raise
        warnings.warn(
            f"visbench.datasets.{_m.name} failed to import "
            f"({type(_e).__name__}: {_e}); datasets in that module are unavailable. "
            f"Set VISBENCH_STRICT_IMPORT=1 to make this fatal.",
            stacklevel=2,
        )
