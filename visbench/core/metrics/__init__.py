"""Metric primitives — pose, homography, classical (Mikolajczyk-style)."""

from .pose import (
    angle_error_mat,
    angle_error_vec,
    compute_pose_error,
    compute_relative_pose,
    estimate_pose_essential,
    pose_auc,
)
from .homography import (
    estimate_homography,
    homography_corner_error,
    hpatches_auc,
)
from .classical import (
    matching_score,
    mma_homography,
    repeatability,
    symmetric_epipolar_distance,
    warp_points,
)

__all__ = [
    "angle_error_mat",
    "angle_error_vec",
    "compute_pose_error",
    "compute_relative_pose",
    "estimate_pose_essential",
    "pose_auc",
    "estimate_homography",
    "homography_corner_error",
    "hpatches_auc",
    "matching_score",
    "mma_homography",
    "repeatability",
    "symmetric_epipolar_distance",
    "warp_points",
]
