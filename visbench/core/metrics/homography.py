"""Homography corner-error metrics for HPatches."""

import cv2
import numpy as np

from ._auc import auc_at_thresholds


def homography_corner_error(H_est: np.ndarray, H_gt: np.ndarray, w: int, h: int) -> float:
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    warped_est = cv2.perspectiveTransform(corners, H_est)[:, 0]
    warped_gt = cv2.perspectiveTransform(corners, H_gt)[:, 0]
    return float(np.linalg.norm(warped_est - warped_gt, axis=1).mean())


def estimate_homography(kpts0: np.ndarray, kpts1: np.ndarray, reproj_thresh: float):
    if len(kpts0) < 4:
        return None
    H, _ = cv2.findHomography(kpts0, kpts1, cv2.USAC_MAGSAC, reproj_thresh, 0.99999, 10000)
    return H


def hpatches_auc(errors: np.ndarray, thresholds: list) -> list:
    return auc_at_thresholds(errors, thresholds)
