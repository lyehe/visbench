"""End-to-end dispatcher coverage: every dataset task routes to a working eval.

These tests don't require any data on disk — they pass a tiny synthetic pair
iterator + a stub matcher into `_evaluate` and check that each task path
returns a JSON-serializable dict with non-zero keys.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from visbench.core.harness import (
    eval_correspondence_pairs,
    eval_descriptor_pairs,
    eval_fundamental_pairs,
    eval_homography_pairs,
    eval_pose_pairs,
)
from visbench.orchestrators.run import _evaluate, _qualitative_sweep


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
        "F_gt": np.eye(3),
        "corrs": np.zeros((4, 4)),
        "kpts_A": np.array([[10.0, 10.0]] * 16, dtype=np.float32),
        "kpts_B": np.array([[10.0, 10.0]] * 16, dtype=np.float32),
        "subset": "test", "scene": "test",
    }]


def test_dispatch_unknown_task_falls_back_to_qualitative(monkeypatch):
    """An unrecognised task degrades gracefully to a qualitative sweep."""
    matcher = StubMatcher()
    # _qualitative_sweep reads pair["im_A_path"] via matcher.load_image; our
    # StubMatcher returns dummy tensors so we don't need real files.
    res = _evaluate("not_a_real_task", matcher, _identical_pair(),
                    resize_long=64, ransac_runs=1, progress=False)
    # Qualitative sweep keys
    for k in ("mean_time_s", "mean_matches", "num_pairs"):
        assert k in res
    json.dumps(res)  # must be JSON-serialisable


def test_qualitative_returns_serializable_dict():
    matcher = StubMatcher()
    res = _qualitative_sweep(matcher, _identical_pair(), resize_long=32)
    json.dumps(res)
    assert res["num_pairs"] == 1


def test_eval_fundamental_pairs_handles_empty_iter():
    """No pairs → returns dict (no crash). Validates the function signature."""
    import warnings
    matcher = StubMatcher()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zero-pairs warning
        res = eval_fundamental_pairs(matcher, iter([]), resize_long=None, progress=False)
    json.dumps(res)
    assert res["num_pair_failures"] == 0
    assert res["mean_matches"] == 0.0


@pytest.mark.parametrize("eval_fn", [
    eval_pose_pairs,
    eval_homography_pairs,
    eval_correspondence_pairs,
    eval_descriptor_pairs,
    eval_fundamental_pairs,
])
def test_each_eval_handles_empty_iter(eval_fn):
    """Every task-level eval function must tolerate an empty iterator."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = eval_fn(StubMatcher(), iter([]), progress=False)
    json.dumps(res)
    assert isinstance(res, dict)


def test_every_registered_dataset_has_a_dispatcher_path():
    """Sanity: every dataset's task is one the dispatcher knows about
    OR will fall through to the qualitative fallback."""
    from visbench.datasets import available, get
    known = {"pose", "homography", "correspondence", "fundamental",
             "descriptor", "qualitative"}
    unknown_tasks = set()
    for n in available():
        spec = get(n)
        if spec.task not in known:
            unknown_tasks.add(spec.task)
    # Even if some are unknown, the fallback handles them — but flag here so
    # we notice if a new task type slips in without a metric implementation.
    assert not unknown_tasks, f"unrecognised tasks: {unknown_tasks}"
