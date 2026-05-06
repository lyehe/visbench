"""Dataset plugin registry — each module self-registers at import time."""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from ._registry import DatasetSpec, available, get, register  # noqa: F401

_pkg_dir = Path(__file__).parent
for _m in pkgutil.iter_modules([str(_pkg_dir)]):
    if _m.name.startswith("_"):
        continue
    importlib.import_module(f"{__name__}.{_m.name}")
