"""Method registry. A method is a callable `build(device) -> matcher`."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable


@dataclass
class MethodSpec:
    name: str
    builder: Callable  # (device: str) -> matcher
    kind: str = "classical"  # informational: "classical" | "vismatch"


_REGISTRY: dict[str, MethodSpec] = {}


def register(name: str, builder: Callable, kind: str = "classical") -> None:
    if name in _REGISTRY:
        warnings.warn(f"Method '{name}' already registered; overwriting.", stacklevel=2)
    _REGISTRY[name] = MethodSpec(name=name, builder=builder, kind=kind)


def get(name: str) -> MethodSpec:
    # vismatch:foo always resolved at lookup time, not registration.
    if name.startswith("vismatch:"):
        from .vismatch_loader import build_vismatch
        sub = name.split(":", 1)[1]
        return MethodSpec(name=name, builder=lambda dev, _n=sub: build_vismatch(_n, dev),
                          kind="vismatch")
    if name not in _REGISTRY:
        raise KeyError(f"Unknown method '{name}'. Registered: {sorted(_REGISTRY)}; "
                       "or use 'vismatch:<name>'.")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
