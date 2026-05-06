"""Pose-estimation metrics — mirrors the RoMa/LoFTR convention."""

import cv2
import numpy as np


def angle_error_mat(R1: np.ndarray, R2: np.ndarray) -> float:
    cos = (np.trace(R1.T @ R2) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    return float(np.rad2deg(np.abs(np.arccos(cos))))


def angle_error_vec(v1: np.ndarray, v2: np.ndarray) -> float:
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    if n < 1e-12:
        return 90.0
    return float(np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))


def compute_relative_pose(R1, t1, R2, t2):
    R = R2 @ R1.T
    t = -R @ t1 + t2
    return R, t


def compute_pose_error(R_est: np.ndarray, t_est: np.ndarray, R_gt: np.ndarray, t_gt: np.ndarray) -> tuple[float, float]:
    e_t = angle_error_vec(t_est.squeeze(), t_gt)
    e_t = min(e_t, 180.0 - e_t)
    e_R = angle_error_mat(R_est, R_gt)
    return e_t, e_R


def estimate_pose_essential(kpts0: np.ndarray, kpts1: np.ndarray, K0: np.ndarray, K1: np.ndarray,
                            norm_thresh: float, conf: float = 0.99999):
    if len(kpts0) < 5:
        return None
    K0inv = np.linalg.inv(K0[:2, :2])
    K1inv = np.linalg.inv(K1[:2, :2])
    k0 = (K0inv @ (kpts0 - K0[None, :2, 2]).T).T
    k1 = (K1inv @ (kpts1 - K1[None, :2, 2]).T).T
    E, mask = cv2.findEssentialMat(k0, k1, np.eye(3), threshold=norm_thresh, prob=conf)
    if E is None:
        return None
    best = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, _ = cv2.recoverPose(_E, k0, k1, np.eye(3), 1e9, mask=mask)
        if n > best:
            best = n
            ret = (R, t, mask.ravel() > 0)
    return ret


def pose_auc(errors: np.ndarray, thresholds: list) -> list:
    errors = np.asarray(errors)
    sort_idx = np.argsort(errors)
    errors = errors[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    aucs = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.r_[recall[:last], recall[last - 1] if last > 0 else 0.0]
        e = np.r_[errors[:last], t]
        aucs.append(float(np.trapezoid(r, x=e) / t))
    return aucs
