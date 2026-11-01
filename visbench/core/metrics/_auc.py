"""Shared AUC primitive — used by both pose-error and homography-corner-error metrics.

Both metrics build a cumulative-recall curve over sorted errors and integrate
under it up to a set of thresholds. The pose and homography variants were
byte-identical; this module centralizes the implementation.
"""

from __future__ import annotations

import numpy as np


def auc_at_thresholds(errors: np.ndarray, thresholds: list) -> list[float]:
    """Trapezoidal area under cumulative-recall(error) curve up to each threshold.

    Returns one AUC per threshold, normalized by the threshold so values are in [0, 1].
    An empty `errors` array returns zeros (so callers can pass an empty result
    without branching).
    """
    errors = np.asarray(errors)
    if errors.size == 0:
        return [0.0 for _ in thresholds]
    errors = np.sort(errors)
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    out: list[float] = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.r_[recall[:last], recall[last - 1] if last > 0 else 0.0]
        e = np.r_[errors[:last], t]
        out.append(float(np.trapezoid(r, x=e) / t))
    return out
