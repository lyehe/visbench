"""Dataset and method registries: loading, duplicate / typo handling, builder checks. No data needed."""

from __future__ import annotations

from pathlib import Path

import pytest

from visbench.datasets._registry import DatasetSpec, _REGISTRY as _DS_REG, register as register_ds
from visbench.methods._registry import register as register_m


def test_dataset_registry_loads():
    from visbench.datasets import available, get
    names = available()
    assert len(names) >= 25, f"expected >=25 datasets, got {len(names)}"
    for n in ("hpatches", "megadepth1500", "wxbs", "eth3d", "minima_metu_vistir"):
        assert n in names, n
        spec = get(n)
        assert callable(spec.iter_pairs)
        assert spec.task in {"pose", "homography", "homography_points", "correspondence"}


def test_method_registry_loads():
    from visbench.methods import available, get
    names = available()
    for n in ("sift", "rootsift", "orb", "akaze", "brisk"):
        assert n in names, n
    spec = get("sift")
    assert callable(spec.builder)


def test_unknown_dataset_raises():
    from visbench.datasets import get
    with pytest.raises(KeyError):
        get("does-not-exist")


def test_unknown_method_raises():
    from visbench.methods import get
    with pytest.raises(KeyError):
        get("definitely-not-a-method")


def test_vismatch_method_routing():
    """`vismatch:foo` should resolve at lookup, even if vismatch isn't installed.

    The lookup itself succeeds; only calling the builder triggers the import.
    """
    from visbench.methods import get
    spec = get("vismatch:something")
    assert spec.kind == "vismatch"


def test_dataset_duplicate_raises():
    spec = DatasetSpec(
        task="homography", iter_pairs=lambda *a, **k: iter([]),
        default_root=Path("x"), source_url="x",
    )
    name = next(iter(_DS_REG))  # any already-registered name
    with pytest.raises(ValueError, match="already registered"):
        register_ds(name, spec)


def test_dataset_unknown_suggests_typo():
    from visbench.datasets import get
    with pytest.raises(KeyError) as exc:
        get("hpataches")  # typo for hpatches
    assert "Did you mean" in str(exc.value)


def test_method_duplicate_raises():
    def builder(device="cpu"):
        return None
    register_m("__test_dup", builder)
    with pytest.raises(ValueError, match="already registered"):
        register_m("__test_dup", builder)


def test_method_builder_missing_device_rejected():
    def bad_builder(weights="x"):  # no device, no **kwargs
        return None
    with pytest.raises(ValueError, match="device"):
        register_m("__bad_builder", bad_builder)


def test_method_builder_var_kw_accepted():
    """**kwargs is enough — it'll catch a `device` keyword."""
    def varkw_builder(**kw):
        return None
    register_m("__varkw_builder", varkw_builder)


def test_task_dispatch_and_valid_tasks_agree():
    """Regression: VALID_TASKS and _TASK_DISPATCH must stay in sync. Adding a
    task to one without the other previously caused register() to accept the
    spec at startup while every cell silently fell to the qualitative fallback
    at eval time."""
    from visbench._tasks import TASKS
    from visbench.orchestrators.run import _TASK_DISPATCH
    from visbench.datasets._registry import VALID_TASKS
    assert frozenset(_TASK_DISPATCH) == TASKS == VALID_TASKS


def test_dataset_get_typo_includes_dataset_count():
    from visbench.datasets import get
    with pytest.raises(KeyError) as exc:
        get("definitely-not-a-real-dataset-name")
    msg = str(exc.value)
    # Should suggest using `visbench list datasets` to see options
    assert "visbench list datasets" in msg
