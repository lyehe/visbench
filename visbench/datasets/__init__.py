"""Dataset plugin registry — each module self-registers at import time.

Loaders are grouped by purpose:

  pose/            relative pose AUC (MegaDepth-1500 and variants, ETH3D, Aachen)
  correspondence/  hand-annotated correspondences, WxBS protocol (WxBS, HardMatch)
  homography/      homography accuracy (HPatches family, SatAst)
  crossmodal/      MINIMA cross-modal suite
  stress/          compact synthetic degradations of HPatches

Shared helpers live in underscore modules here (`_registry`, `_download`,
`_rotation`) and inside each group. Module-import errors are collected into
`IMPORT_ERRORS` rather than raising during package import, so one broken
plugin cannot hide the others; they are also warned once. Set
VISBENCH_STRICT_IMPORT=1 to make any plugin failure fatal.
"""

from __future__ import annotations

import importlib
import os
import pkgutil
import warnings
from pathlib import Path

from ._registry import DatasetSpec, available, get, register  # noqa: F401

IMPORT_ERRORS: dict[str, str] = {}

GROUPS = ("pose", "correspondence", "homography", "crossmodal", "stress")

_pkg_dir = Path(__file__).parent
_strict = os.environ.get("VISBENCH_STRICT_IMPORT") == "1"
for _group in GROUPS:
    for _m in pkgutil.iter_modules([str(_pkg_dir / _group)]):
        if _m.name.startswith("_"):
            continue
        _qual = f"{__name__}.{_group}.{_m.name}"
        try:
            importlib.import_module(_qual)
        except Exception as _e:  # noqa: BLE001
            IMPORT_ERRORS[f"{_group}.{_m.name}"] = f"{type(_e).__name__}: {_e}"
            if _strict:
                raise
            warnings.warn(
                f"{_qual} failed to import ({type(_e).__name__}: {_e}); datasets in that "
                f"module are unavailable. Set VISBENCH_STRICT_IMPORT=1 to make this fatal.",
                stacklevel=2,
            )
