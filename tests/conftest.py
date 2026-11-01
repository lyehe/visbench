"""Test bootstrap.

If `torch` or `vismatch` are missing (e.g. a CI image that intentionally drops
the GPU stack for structural-only test runs), install minimal stubs so the
visbench import chain still succeeds. With the real packages installed this
file is a no-op.

This keeps `pytest` runnable on a CPU/CI box that hasn't paid the cost of
installing torch + vismatch + all per-matcher submodules, which is the whole
point of having a "structural" test tier separate from real benchmarks.
"""

from __future__ import annotations

import sys
import types


def _importable(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def _stub_torch() -> None:
    if "torch" in sys.modules or _importable("torch"):
        return
    t = types.ModuleType("torch")

    class _Tensor:
        pass

    t.Tensor = _Tensor
    sys.modules["torch"] = t


def _stub_vismatch() -> None:
    if "vismatch" in sys.modules or _importable("vismatch"):
        return
    vm = types.ModuleType("vismatch")
    vm.available_models = []
    vm.MATCHER_NAMES = []

    def _get_matcher(*a, **k):
        raise RuntimeError("vismatch not installed (test stub)")

    vm.get_matcher = _get_matcher
    vm.get_default_device = lambda: "cpu"
    sys.modules["vismatch"] = vm

    bm = types.ModuleType("vismatch.base_matcher")

    class _BaseMatcher:
        def __init__(self, device: str = "cpu", **kw):
            self.device = device

        def load_image(self, path, resize=None):
            return None

        def __call__(self, img0, img1):
            return {}

    bm.BaseMatcher = _BaseMatcher
    sys.modules["vismatch.base_matcher"] = bm


_stub_torch()
_stub_vismatch()
