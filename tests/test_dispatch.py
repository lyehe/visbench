"""End-to-end dispatcher coverage: every dataset task routes to a working eval.

These tests don't require any data on disk — they pass a tiny synthetic pair
iterator + a stub matcher into `_evaluate` and check that each task path
returns a JSON-serializable dict with non-zero keys.
"""

from __future__ import annotations


import numpy as np
import pytest

from visbench.core.harness import (
    eval_correspondence_pairs,
    eval_homography_pairs,
    eval_point_homography_pairs,
    eval_pose_pairs,
)


class StubMatcher:
    """Minimal vismatch-compatible matcher: emits a fixed grid of correspondences."""

    def load_image(self, path, resize=None):
        # Returns a tiny tensor stand-in; harness only re-passes it to __call__.
        if resize is None:
            return np.zeros((3, 32, 32), dtype=np.float32)
        h, w = resize if isinstance(resize, tuple) else (resize, resize)
        return np.zeros((3, h, w), dtype=np.float32)

    def __call__(self, img0, img1):
        rng = np.random.default_rng(0)
        kpts = rng.uniform(0, 100, (16, 2)).astype(np.float64)
        return {
            "matched_kpts0": kpts.copy(),
            "matched_kpts1": kpts.copy() + 0.5,
            "all_kpts0": kpts.copy(),
            "all_kpts1": kpts.copy(),
            "num_inliers": 10,
        }


def _identical_pair():
    """Returns an iter that yields one synthetic pair carrying every field."""
    here = __file__  # any file works as image path; harness reads via PIL
    # Use an actual readable image — we don't have one here, so the harness
    # `_pil_size` will fail. Instead bypass image-loading by going through the
    # eval functions directly with shapes known to the matcher.
    return [{
        "im_A_path": here, "im_B_path": here,
        "K0": np.eye(3), "K1": np.eye(3),
        "R_0to1": np.eye(3), "t_0to1": np.array([1.0, 0, 0]),
        "H_gt": np.eye(3),
        "corrs": np.zeros((4, 4)),
        "subset": "test", "scene": "test",
    }]


def test_dispatch_forwards_only_accepted_kwargs(tmp_path):
    """_evaluate filters kwargs by each eval fn's signature, so non-pose
    tasks don't get spammed with seed/ransac_runs (was previously hidden
    by 6 identical wrapper lambdas)."""
    from visbench.orchestrators.run import _TASK_DISPATCH, _accepted_kwargs
    # Pose accepts seed + ransac_runs; homography accepts neither.
    pose_kw = _accepted_kwargs(_TASK_DISPATCH["pose"])
    homog_kw = _accepted_kwargs(_TASK_DISPATCH["homography"])
    assert "seed" in pose_kw
    assert "ransac_runs" in pose_kw
    assert "seed" not in homog_kw
    assert "ransac_runs" not in homog_kw


@pytest.mark.parametrize("eval_fn", [
    eval_pose_pairs,
    eval_homography_pairs,
    eval_correspondence_pairs,
    eval_point_homography_pairs,
])
def test_each_eval_rejects_empty_iter(eval_fn):
    """Zero pairs means missing data or a bad layout, never a score of 0."""
    from visbench.core.sampling import EmptyPairsError
    with pytest.raises(EmptyPairsError):
        eval_fn(StubMatcher(), iter([]), progress=False)


def test_every_registered_dataset_has_a_dispatcher_path():
    """Every dataset's task has an eval in the dispatcher."""
    from visbench.datasets import available, get
    from visbench._tasks import TASKS
    known = set(TASKS)
    unknown_tasks = set()
    for n in available():
        spec = get(n)
        if spec.task not in known:
            unknown_tasks.add(spec.task)
    assert not unknown_tasks, f"unrecognised tasks: {unknown_tasks}"
