"""Registry round-trip tests — fast, no data download."""

from __future__ import annotations

import pytest


def test_dataset_registry_loads():
    from visbench.datasets import available, get
    names = available()
    assert len(names) > 30, f"expected >=30 datasets, got {len(names)}"
    for n in ("hpatches", "oxford_affine", "evd", "megadepth1500", "tum_rgbd", "blendedmvs"):
        assert n in names, n
        spec = get(n)
        assert callable(spec.iter_pairs)
        assert spec.task in {"pose", "homography", "correspondence", "descriptor",
                             "fundamental", "qualitative"}


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
