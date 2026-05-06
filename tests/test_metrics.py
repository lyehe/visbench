"""Sanity checks on metric primitives — no data needed."""

import numpy as np

from visbench.core.metrics.pose import (
    angle_error_mat, angle_error_vec, compute_pose_error, pose_auc,
)
from visbench.core.metrics.homography import (
    estimate_homography, homography_corner_error, hpatches_auc,
)
from visbench.core.metrics.classical import (
    mma_homography, repeatability, warp_points,
)


def test_angle_error_identity():
    R = np.eye(3)
    assert angle_error_mat(R, R) < 1e-6


def test_angle_error_vec_parallel():
    v = np.array([1.0, 2.0, 3.0])
    assert angle_error_vec(v, v) < 1e-6


def test_compute_pose_error_zero():
    R, t = np.eye(3), np.array([1.0, 0.0, 0.0])
    e_t, e_R = compute_pose_error(R, t, R, t)
    assert e_t < 1e-6 and e_R < 1e-6


def test_pose_auc_perfect():
    errs = np.zeros(100)
    aucs = pose_auc(errs, [5, 10, 20])
    for a in aucs:
        assert a > 0.99


def test_pose_auc_all_failed():
    errs = np.full(100, 90.0)
    aucs = pose_auc(errs, [5, 10, 20])
    for a in aucs:
        assert a < 0.01


def test_homography_corner_error_identity():
    H = np.eye(3)
    err = homography_corner_error(H, H, w=320, h=240)
    assert err < 1e-6


def test_estimate_homography_identity():
    rng = np.random.default_rng(0)
    pts0 = rng.uniform(0, 100, (50, 2)).astype(np.float64)
    H = estimate_homography(pts0, pts0, reproj_thresh=1.0)
    assert H is not None
    err = homography_corner_error(H, np.eye(3), w=200, h=200)
    assert err < 1e-3


def test_warp_points_identity():
    pts = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = warp_points(pts, np.eye(3))
    assert np.allclose(out, pts)


def test_mma_perfect():
    pts = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    res = mma_homography(pts, pts, np.eye(3))
    assert res["mma_1"] > 0.99


def test_repeatability_overlap():
    kpts0 = np.array([[10.0, 10.0], [50.0, 50.0]])
    res = repeatability(kpts0, kpts0, np.eye(3), image_shape1=(100, 100), px_thresh=2.0)
    assert res["repeatability"] > 0.99
