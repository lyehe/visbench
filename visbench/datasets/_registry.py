"""Dataset registry primitives — no auto-import side-effects."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

VALID_TASKS = {"pose", "homography", "correspondence", "descriptor",
               "fundamental", "qualitative"}


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
    if spec.task not in VALID_TASKS:
        raise ValueError(
            f"Dataset '{name}' has unknown task '{spec.task}'. "
            f"Expected one of: {sorted(VALID_TASKS)}"
        )
    if name in _REGISTRY:
        warnings.warn(
            f"Dataset '{name}' already registered; previous spec will be overwritten.",
            stacklevel=2,
        )
    _REGISTRY[name] = spec


def get(name: str) -> DatasetSpec:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
