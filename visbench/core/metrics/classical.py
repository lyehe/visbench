"""Classical detector/descriptor metrics (Mikolajczyk-style).

Cleaner signal than pose AUC when iterating on handcrafted features.
Applicable where ground-truth correspondence is direct (HPatches: GT homography).
For pose benchmarks (MegaDepth/ScanNet), correctness uses epipolar distance
rather than direct reprojection.
"""

from __future__ import annotations

import numpy as np


def warp_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Apply 3x3 homography to (N,2) points."""
    if len(pts) == 0:
        return pts
    ones = np.ones((len(pts), 1), dtype=pts.dtype)
    hom = np.concatenate([pts, ones], axis=1) @ H.T
    return hom[:, :2] / hom[:, 2:3]


def repeatability(kpts0: np.ndarray, kpts1: np.ndarray, H_0to1: np.ndarray,
                  image_shape1: tuple, px_thresh: float = 3.0) -> dict:
    """Fraction of kpts from img0 that have a neighbor in img1 within px_thresh
    after warping by H_0to1. Keeps only kpts visible in img1.
    """
    if len(kpts0) == 0 or len(kpts1) == 0:
        return {"repeatability": 0.0, "n_overlap": 0}
    h, w = image_shape1[:2]
    warped = warp_points(kpts0, H_0to1)
    vis = (warped[:, 0] >= 0) & (warped[:, 0] < w) & (warped[:, 1] >= 0) & (warped[:, 1] < h)
    warped_v = warped[vis]
    if len(warped_v) == 0:
        return {"repeatability": 0.0, "n_overlap": 0}
    dists = np.linalg.norm(warped_v[:, None, :] - kpts1[None, :, :], axis=2).min(axis=1)
    return {
        "repeatability": float((dists < px_thresh).mean()),
        "n_overlap": int(vis.sum()),
    }


def mma_homography(matched_kpts0: np.ndarray, matched_kpts1: np.ndarray,
                   H_0to1: np.ndarray, thresholds=(1, 2, 3, 5, 10)) -> dict:
    """Mean Matching Accuracy: fraction of putative matches whose reprojection
    error under GT H is below each threshold. Standard in D2-Net/SuperPoint
    papers for HPatches detector+descriptor evaluation.
    """
    if len(matched_kpts0) == 0:
        return {f"mma_{int(t)}": 0.0 for t in thresholds} | {"n_matches": 0}
    warped = warp_points(matched_kpts0, H_0to1)
    err = np.linalg.norm(warped - matched_kpts1, axis=1)
    out = {f"mma_{int(t)}": float((err < t).mean()) for t in thresholds}
    out["n_matches"] = int(len(matched_kpts0))
    out["mean_err_px"] = float(err.mean())
    return out


def matching_score(matched_kpts0: np.ndarray, matched_kpts1: np.ndarray,
                   all_kpts0: np.ndarray, H_0to1: np.ndarray,
                   image_shape1: tuple, px_thresh: float = 3.0) -> float:
    """Fraction of detected kpts that became *correct* matches.
    Correct = reprojection error under GT H is below px_thresh AND kpt is visible.
    """
    if len(matched_kpts0) == 0:
        return 0.0
    warped = warp_points(matched_kpts0, H_0to1)
    h, w = image_shape1[:2]
    vis = (warped[:, 0] >= 0) & (warped[:, 0] < w) & (warped[:, 1] >= 0) & (warped[:, 1] < h)
    err = np.linalg.norm(warped - matched_kpts1, axis=1)
    correct = vis & (err < px_thresh)
    denom = max(len(all_kpts0), len(matched_kpts0), 1)
    return float(correct.sum() / denom)


def symmetric_epipolar_distance(kpts0: np.ndarray, kpts1: np.ndarray,
                                K0: np.ndarray, K1: np.ndarray,
                                R_0to1: np.ndarray, t_0to1: np.ndarray) -> np.ndarray:
    """Sampson / symmetric epipolar distance in pixels for pose benchmarks."""
    if len(kpts0) == 0:
        return np.empty((0,))
    t = t_0to1.reshape(3)
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = tx @ R_0to1
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
    h0 = np.concatenate([kpts0, np.ones((len(kpts0), 1))], axis=1)
    h1 = np.concatenate([kpts1, np.ones((len(kpts1), 1))], axis=1)
    Fx0 = (F @ h0.T).T
    Fx1 = (F.T @ h1.T).T
    num = (h1 * Fx0).sum(axis=1) ** 2
    denom = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2
    return np.sqrt(np.clip(num / np.clip(denom, 1e-12, None), 0, None))
