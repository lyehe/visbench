"""Pair-dict schemas — typed contracts for what `iter_pairs` yields per task.

These TypedDicts document the keys each eval_* function consumes. They are not
runtime-enforced (each iter_pairs is free to yield extra keys), but they:

  1. Give a single place to grep when adding a new dataset / new task.
  2. Are reflected in `REQUIRED_KEYS_BY_TASK` for the per-benchmark smoke test
     that checks yielded pairs have the right shape when real data is on disk.

Adding a new task: add a new TypedDict here, add its required keys to
REQUIRED_KEYS_BY_TASK, and add a `_TASK_DISPATCH` entry in orchestrators/run.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    import numpy as np  # noqa: F401  # only for type hints below


class _Base(TypedDict, total=False):
    im_A_path: str
    im_B_path: str
    subset: str
    scene: str


class PosePair(_Base, total=False):
    """`task=pose`. K0/K1 are 3x3 intrinsics, R/t the GT relative pose A->B."""
    K0: "np.ndarray"
    K1: "np.ndarray"
    R_0to1: "np.ndarray"
    t_0to1: "np.ndarray"
    resize_hw: tuple[int, int]
    K_already_scaled: bool


class HomographyPair(_Base, total=False):
    """`task=homography`. H_gt is 3x3 (img-A -> img-B coords)."""
    H_gt: "np.ndarray"


class CorrespondencePair(_Base, total=False):
    """`task=correspondence`. corrs is (N, 4): [x0, y0, x1, y1] per GT match."""
    corrs: "np.ndarray"


class FundamentalPair(_Base, total=False):
    """`task=fundamental`. F_gt is the 3x3 fundamental matrix."""
    F_gt: "np.ndarray"


class DescriptorPair(_Base, total=False):
    """`task=descriptor`. Anchor and target kpts; sizes optional."""
    kpts_A: "np.ndarray"
    kpts_B: "np.ndarray"
    kpt_sizes_A: "np.ndarray"
    kpt_sizes_B: "np.ndarray"


REQUIRED_KEYS_BY_TASK: dict[str, tuple[str, ...]] = {
    "pose":           ("im_A_path", "im_B_path", "K0", "K1", "R_0to1", "t_0to1"),
    "homography":     ("im_A_path", "im_B_path", "H_gt"),
    "correspondence": ("im_A_path", "im_B_path", "corrs"),
    "fundamental":    ("im_A_path", "im_B_path", "F_gt"),
    "descriptor":     ("im_A_path", "im_B_path", "kpts_A", "kpts_B"),
    "qualitative":    ("im_A_path", "im_B_path"),
}


__all__ = [
    "PosePair", "HomographyPair", "CorrespondencePair",
    "FundamentalPair", "DescriptorPair",
    "REQUIRED_KEYS_BY_TASK",
]
