"""Tests for the matcher protocol surface introduced during the audit.

These verify that classical matchers expose `get_cv_feature()` returning their
underlying cv2 object — required by the descriptor-isolation eval path so it
no longer reaches into private `_sift`/`_orb` attributes.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from visbench.methods.classical import (
    AKAZEFeature, BRISKFeature, ORBFeature, RootSIFTFeature, SIFTFeature,
)


@pytest.mark.parametrize("cls", [SIFTFeature, RootSIFTFeature, ORBFeature,
                                  AKAZEFeature, BRISKFeature])
def test_get_cv_feature_returns_compute_capable(cls):
    f = cls(device="cpu").get_cv_feature()
    assert f is not None
    assert hasattr(f, "compute"), \
        f"{cls.__name__}.get_cv_feature() returned {f!r} without .compute()"


def test_rootsift_normalize_is_l2_unit():
    """RootSIFT descriptors after L1-norm-then-sqrt are L2-unit."""
    from visbench.methods.classical import _rootsift_normalize
    raw = np.random.default_rng(0).uniform(0, 100, (8, 128)).astype(np.float32)
    out = _rootsift_normalize(raw)
    norms = np.linalg.norm(out, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


def test_rootsift_normalize_handles_empty():
    from visbench.methods.classical import _rootsift_normalize
    out = _rootsift_normalize(np.empty((0, 128), dtype=np.float32))
    assert out.shape == (0, 128)


def test_resolve_cv_feature_via_protocol():
    """The internal resolver should prefer get_cv_feature() over attribute sniff."""
    from visbench.core.harness import _resolve_cv_feature
    m = SIFTFeature(device="cpu")
    f = _resolve_cv_feature(m)
    assert f is m._sift  # both paths reach the same object


def test_resolve_cv_feature_legacy_fallback():
    """Matchers without get_cv_feature still resolve via _sift/_orb sniff."""
    from visbench.core.harness import _resolve_cv_feature

    class LegacyMatcher:
        def __init__(self):
            self._sift = cv2.SIFT_create(nfeatures=8)

    m = LegacyMatcher()
    f = _resolve_cv_feature(m)
    assert f is m._sift
