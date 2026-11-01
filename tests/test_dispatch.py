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


def _identical_pair_with_real_image(tmp_path):
    """Like _identical_pair() but writes an actual image file so PIL.Image.open
    succeeds — needed since _qualitative_sweep now goes through the same
    aspect-preserving resize as the other eval_* paths."""
    from PIL import Image
    import numpy as np
    p = tmp_path / "stub.png"
    Image.new("RGB", (32, 32)).save(p)
    pair = dict(_identical_pair()[0])
    pair["im_A_path"] = str(p)
    pair["im_B_path"] = str(p)
    return [pair]


def test_dispatch_unknown_task_falls_back_to_qualitative(tmp_path):
    """An unrecognised task degrades gracefully to a qualitative sweep."""
    matcher = StubMatcher()
    res = _evaluate("not_a_real_task", matcher,
                    _identical_pair_with_real_image(tmp_path),
                    resize_long=64, ransac_runs=1, progress=False, seed=0)
    for k in ("mean_time_s", "mean_matches", "num_pairs"):
        assert k in res
    json.dumps(res)  # must be JSON-serialisable


def test_qualitative_returns_serializable_dict(tmp_path):
    matcher = StubMatcher()
    res = _qualitative_sweep(matcher, _identical_pair_with_real_image(tmp_path),
                             resize_long=32)
    json.dumps(res)
    assert res["num_pairs"] == 1


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


def test_qualitative_warns_on_pair_failure(tmp_path):
    """Regression: _qualitative_sweep used to silently swallow per-pair
    exceptions; now it emits the same UserWarning the eval_*_pairs paths do."""
    import warnings

    class CrashingMatcher(StubMatcher):
        def __call__(self, img0, img1):
            raise RuntimeError("simulated matcher crash")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = _qualitative_sweep(CrashingMatcher(),
                                 _identical_pair_with_real_image(tmp_path),
                                 resize_long=32)
    assert res["num_pair_failures"] == 1
    assert any("_qualitative_sweep" in str(w.message) for w in caught), \
        f"expected per-pair warning, got: {[str(w.message) for w in caught]}"


@pytest.mark.parametrize("eval_fn", [
    eval_pose_pairs,
    eval_homography_pairs,
    eval_correspondence_pairs,
    eval_descriptor_pairs,
    eval_fundamental_pairs,
])
def test_each_eval_rejects_empty_iter(eval_fn):
    """Zero pairs means missing data or a bad layout, never a score of 0."""
    from visbench.core.sampling import EmptyPairsError
    with pytest.raises(EmptyPairsError):
        eval_fn(StubMatcher(), iter([]), progress=False)


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
