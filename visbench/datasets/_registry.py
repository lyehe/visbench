"""Dataset registry primitives — no auto-import side-effects."""

from __future__ import annotations

import difflib
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# VALID_TASKS aliases the canonical set in `visbench._tasks` so the dataset
# registry and the orchestrator's `_TASK_DISPATCH` never drift.
from .._tasks import TASKS as VALID_TASKS  # noqa: E402  (intentional)


@dataclass
class DatasetSpec:
    task: str
    iter_pairs: Callable
    default_root: Path
    default_resize: int | None = None
    kwargs: dict | None = None
    download: Callable | None = None
    source_url: str = ""
    license: str = ""


_REGISTRY: dict[str, DatasetSpec] = {}


def register(name: str, spec: DatasetSpec) -> None:
    """Register a dataset. Duplicates raise unless VISBENCH_ALLOW_REREGISTER=1.

    Silent clobbering hides bugs where two modules ship the same name; the env
    var exists only for interactive notebook reload workflows.
    """
    if spec.task not in VALID_TASKS:
        raise ValueError(
            f"Dataset '{name}' has unknown task '{spec.task}'. "
            f"Expected one of: {sorted(VALID_TASKS)}"
        )
    if name in _REGISTRY:
        if os.environ.get("VISBENCH_ALLOW_REREGISTER") == "1":
            warnings.warn(
                f"Dataset '{name}' re-registered (VISBENCH_ALLOW_REREGISTER=1).",
                stacklevel=2,
            )
        else:
            raise ValueError(
                f"Dataset '{name}' is already registered. Duplicate registration "
                f"is almost always a bug (two modules using the same name, or a "
                f"module imported twice). If you're iterating in a notebook, set "
                f"VISBENCH_ALLOW_REREGISTER=1 to allow overwrite."
            )
    _REGISTRY[name] = spec


def get(name: str) -> DatasetSpec:
    if name not in _REGISTRY:
        registered = sorted(_REGISTRY)
        suggestions = difflib.get_close_matches(name, registered, n=3, cutoff=0.6)
        msg = f"Unknown dataset '{name}'."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        msg += (f" {len(registered)} datasets are registered; "
                f"run `visbench list datasets` to see them all.")
        raise KeyError(msg)
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
