"""Method registry. A method is a callable `build(device) -> matcher`."""

from __future__ import annotations

import difflib
import inspect
import os
import warnings
from dataclasses import dataclass
from typing import Callable


@dataclass
class MethodSpec:
    name: str
    builder: Callable  # (device: str) -> matcher
    kind: str = "classical"  # informational: "classical" | "vismatch" | "romaxx"


_REGISTRY: dict[str, MethodSpec] = {}


def _validate_builder_signature(name: str, builder: Callable) -> None:
    """Builder must accept a `device` parameter (positional or keyword).

    Catches a common bug class where a refactor drops the device kwarg and the
    crash only surfaces on first `build(device=...)` call inside a benchmark.
    """
    try:
        sig = inspect.signature(builder)
    except (TypeError, ValueError):
        return  # built-ins, C-callables — skip silently
    if "device" in sig.parameters:
        return
    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD
                     for p in sig.parameters.values())
    if has_var_kw:
        return
    raise ValueError(
        f"Method '{name}' builder {builder!r} does not accept a `device` "
        f"parameter. The harness calls `builder(device=...)`, so all builders "
        f"must accept it (or accept **kwargs)."
    )


def register(name: str, builder: Callable, kind: str = "classical") -> None:
    """Register a method. Duplicates raise unless VISBENCH_ALLOW_REREGISTER=1."""
    _validate_builder_signature(name, builder)
    if name in _REGISTRY:
        if os.environ.get("VISBENCH_ALLOW_REREGISTER") == "1":
            warnings.warn(
                f"Method '{name}' re-registered (VISBENCH_ALLOW_REREGISTER=1).",
                stacklevel=2,
            )
        else:
            raise ValueError(
                f"Method '{name}' is already registered. Set "
                f"VISBENCH_ALLOW_REREGISTER=1 if you really want to overwrite."
            )
    _REGISTRY[name] = MethodSpec(name=name, builder=builder, kind=kind)


def get(name: str) -> MethodSpec:
    # vismatch:foo always resolved at lookup time, not registration.
    if name.startswith("vismatch:"):
        from .vismatch_loader import build_vismatch
        sub = name.split(":", 1)[1]
        return MethodSpec(name=name, builder=lambda dev, _n=sub: build_vismatch(_n, dev),
                          kind="vismatch")
    if name.startswith("romaxx:"):
        from .romaxx_loader import build_romaxx
        return MethodSpec(name=name, builder=lambda dev, _n=name: build_romaxx(_n, dev),
                          kind="romaxx")
    if name not in _REGISTRY:
        registered = sorted(_REGISTRY)
        suggestions = difflib.get_close_matches(name, registered, n=3, cutoff=0.6)
        msg = f"Unknown method '{name}'."
        if suggestions:
            msg += f" Did you mean: {', '.join(suggestions)}?"
        msg += (f" Registered classical methods: {registered}. "
                "Use 'vismatch:<name>' for vismatch matchers or 'romaxx:<kind>[:<arg>]' "
                "for RoMaXX matchers (see visbench/methods/romaxx_loader.py).")
        raise KeyError(msg)
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)
