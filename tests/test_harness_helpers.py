"""Tests for previously-untested harness primitives.

These all run without GPU / data / network — they just exercise the pure
helpers introduced or refactored during the audit pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from visbench.core.harness import (
    _compute_map_topk,
    _epi_distance,
    _scale_K,
    DEFAULT_KPT_SIZE_PX,
    MIN_FUND_MATCHES,
    MIN_HOMOG_MATCHES,
    MIN_POSE_MATCHES,
    POSE_FAIL_DEG,
)
from visbench.core.metrics._auc import auc_at_thresholds


# ----- _scale_K -------------------------------------------------------------

def test_scale_K_independent_axes():
    K = np.array([[100.0, 0.0, 50.0],
                  [0.0, 100.0, 25.0],
                  [0.0, 0.0, 1.0]])
    K2 = _scale_K(K, 2.0, 0.5)
    assert K2[0, 0] == 200.0 and K2[0, 2] == 100.0
    assert K2[1, 1] == 50.0 and K2[1, 2] == 12.5
    assert K2[2, 2] == 1.0


def test_scale_K_does_not_mutate_input():
    K = np.array([[100.0, 0.0, 50.0],
                  [0.0, 100.0, 25.0],
                  [0.0, 0.0, 1.0]])
    K_copy = K.copy()
    _ = _scale_K(K, 3.0, 7.0)
    np.testing.assert_array_equal(K, K_copy)


# ----- _compute_map_topk ----------------------------------------------------

def test_map_topk_perfect_identity():
    """Identical descriptors → rank 0 everywhere → top1=1, mAP=1."""
    d = np.eye(8, dtype=np.float32)
    res = _compute_map_topk(d, d, metric="l2")
    assert res["n"] == 8
    assert res["top1_acc"] == pytest.approx(1.0)
    assert res["top5_acc"] == pytest.approx(1.0)
    assert res["mAP"] == pytest.approx(1.0)


def test_map_topk_hamming_perfect():
    """uint8 descriptors with hamming metric, identity case."""
    d = np.eye(8, dtype=np.uint8)
    # Pack to bytes: 1 column -> bit 0; just ensure rank order is correct.
    d_bytes = np.packbits(d, axis=1)
    res = _compute_map_topk(d_bytes, d_bytes, metric="hamming")
    assert res["top1_acc"] == pytest.approx(1.0)


def test_map_topk_empty_inputs():
    res = _compute_map_topk(np.empty((0, 8)), np.empty((0, 8)))
    assert res == {"top1_acc": 0.0, "top5_acc": 0.0, "mAP": 0.0, "n": 0}


def test_map_topk_mismatched_lengths_returns_zero():
    res = _compute_map_topk(np.eye(4), np.eye(5))
    assert res["n"] == 0


# ----- _epi_distance --------------------------------------------------------

def test_epi_distance_matching_corresponds_to_zero_residual():
    """For the canonical F = [[0,0,0],[0,0,-1],[0,1,0]] (pure translation in y),
    matched points on the same horizontal line have zero epipolar distance."""
    F = np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0],
                  [0.0, 1.0, 0.0]], dtype=np.float64)
    # Points (x, y) and (x, y) on the same horizontal line: the epipolar
    # constraint x1' F x0 = -y + y = 0 is satisfied.
    kpts0 = np.array([[3.0, 5.0], [10.0, 7.0]])
    kpts1 = np.array([[3.0, 5.0], [10.0, 7.0]])
    d = _epi_distance(F, kpts0, kpts1)
    np.testing.assert_allclose(d, 0.0, atol=1e-9)


def test_epi_distance_none_F_returns_inf():
    d = _epi_distance(None, np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]))
    assert d.shape == (1,)
    assert np.isinf(d[0])


def test_epi_distance_empty_input():
    d = _epi_distance(np.eye(3), np.empty((0, 2)), np.empty((0, 2)))
    assert d.shape == (0,)


# ----- auc_at_thresholds ----------------------------------------------------

def test_auc_perfect_zero_errors():
    aucs = auc_at_thresholds(np.zeros(100), [5, 10, 20])
    for a in aucs:
        assert a > 0.99


def test_auc_all_failed():
    aucs = auc_at_thresholds(np.full(100, 90.0), [5, 10, 20])
    for a in aucs:
        assert a < 0.01


def test_auc_empty_returns_zeros():
    aucs = auc_at_thresholds(np.array([]), [5, 10, 20])
    assert aucs == [0.0, 0.0, 0.0]


def test_auc_pose_and_hpatches_are_byte_identical():
    """Regression: pose_auc and hpatches_auc were separately-defined copies
    of the same function. They now share the underlying primitive."""
    from visbench.core.metrics.pose import pose_auc
    from visbench.core.metrics.homography import hpatches_auc
    errs = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0])
    a = pose_auc(errs, [5, 10, 20])
    b = hpatches_auc(errs, [5, 10, 20])
    assert a == b


# ----- threshold constants are consistent with cv2 minimums ----------------

def test_min_match_constants():
    """These are dictated by cv2's algorithm minimums; document them here so a
    refactor doesn't silently raise or lower the floor."""
    assert MIN_POSE_MATCHES == 5      # 5-point essential matrix
    assert MIN_FUND_MATCHES == 8      # 8-point fundamental matrix
    assert MIN_HOMOG_MATCHES == 4     # 4 point pairs for homography
    assert POSE_FAIL_DEG == 90.0
    assert DEFAULT_KPT_SIZE_PX == 8.0
