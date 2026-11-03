"""Classical OpenCV matchers: detect/describe surface and RootSIFT normalization."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from visbench.methods.classical import (
    AKAZEFeature, BRISKFeature, ORBFeature, RootSIFTFeature, SIFTFeature,
)


@pytest.mark.parametrize("cls", [SIFTFeature, RootSIFTFeature, ORBFeature,
                                  AKAZEFeature, BRISKFeature])
def test_classical_matchers_detect_and_describe(cls):
    img = (np.random.default_rng(0).uniform(0, 255, (96, 96))).astype(np.uint8)
    kpts, desc = cls(device="cpu").detect_and_describe(img)
    assert kpts.ndim == 2 and kpts.shape[1] == 2 and len(desc) == len(kpts)


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

