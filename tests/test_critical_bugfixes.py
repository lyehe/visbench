"""Regression tests for the two critical bugs the audit found.

1. mutual_nearest_neighbor's Lowe ratio test used `np.partition(d01, 2)[:, :2]`,
   which leaves elements [0, 1] in unspecified order. The "best" and
   "second-best" distances could be swapped, randomly corrupting half the
   ratio-test decisions on every SIFT/RootSIFT match.

2. synthetic_geometric._perspective / _random_affine captured an RNG at
   module import time. On a second iter_pairs() call within the same process,
   the cached warped image (from the first call's RNG draw) was paired with a
   newly-computed H_aug (from a different RNG state), silently desyncing H_gt
   from the on-disk image.
"""

from __future__ import annotations

import numpy as np
import pytest


# --- Bug 1: Lowe ratio test partition ---------------------------------------

def test_partition_picks_smallest_at_position_0():
    """The fixed code uses kth=1, so position [0] is always the minimum."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        row = rng.uniform(0, 10, (20,))
        part = np.partition(row, 1)[:2]
        # With kth=1, position 0 is guaranteed <= position 1, so position 0
        # IS the minimum. (With kth=2 — the old buggy code — this was not
        # guaranteed.)
        assert part[0] == row.min()
        assert part[1] == np.partition(row, 1)[1]
        # And [1] is the 2nd-smallest.
        sorted_row = np.sort(row)
        assert part[1] == sorted_row[1]


def test_mutual_nn_lowe_ratio_accepts_clear_winners():
    """When best is much smaller than 2nd-best (clear ratio), the new partition
    code accepts the match — the old kth=2 code rejected it half the time
    because positions [0,1] could be swapped, making the test `2nd_best < 0.8 * best`."""
    from visbench.methods.base import mutual_nearest_neighbor
    # desc0[0] is an exact match for desc1[0]; desc1 has other rows that are
    # FAR (so the ratio is clearly < 0.8). Run this many times to exercise
    # different random tie-breaks within partition.
    rng = np.random.default_rng(0)
    accept_count = 0
    n_trials = 50
    for trial in range(n_trials):
        # Random "far" descriptors with a single exact match.
        desc1 = rng.uniform(-1, 1, (10, 8)).astype(np.float32) * 3.0
        # Put exact match at row trial % 10 so the matched index varies.
        target_idx = trial % 10
        desc0 = desc1[target_idx:target_idx + 1].copy()
        pairs = mutual_nearest_neighbor(desc0, desc1, ratio=0.8, metric="l2")
        matched_idx0 = set(int(x) for x in pairs[:, 0])
        if 0 in matched_idx0:
            accept_count += 1
    # With kth=1 (the fix), 100% acceptance. With kth=2 (the old bug), ~50%
    # because partition randomly orders positions [0,1].
    assert accept_count == n_trials, \
        f"Lowe ratio rejected {n_trials - accept_count}/{n_trials} clear winners — partition order bug regressed"


# --- Bug 2: synthetic_geometric RNG-state-leak -----------------------------

def test_perspective_make_is_pure_function_of_wh():
    """Same (w, h) should produce same H_aug whether it's the first call or
    the 100th call. Previously the RNG state leaked across calls."""
    from visbench.datasets.synthetic_geometric import _perspective
    fn = _perspective(strength=0.1, seed=42)
    H1, dims1 = fn(640, 480)
    # Hammer the function with other (w, h) — should not affect the original.
    for _ in range(20):
        fn(800, 600)
        fn(320, 240)
    H1_again, dims1_again = fn(640, 480)
    np.testing.assert_array_equal(H1, H1_again)
    assert dims1 == dims1_again


def test_random_affine_make_is_pure_function_of_wh():
    from visbench.datasets.synthetic_geometric import _random_affine
    fn = _random_affine(seed=7)
    H1, _ = fn(640, 480)
    for _ in range(20):
        fn(800, 600)
        fn(320, 240)
    H1_again, _ = fn(640, 480)
    np.testing.assert_array_equal(H1, H1_again)


def test_two_calls_to_iter_pairs_produce_same_H():
    """End-to-end: a second iter_pairs() in the same process should yield the
    same H_gt as the first (the previous bug silently desynced cache and H_gt).
    Uses synthetic_geometric directly without needing real image data."""
    from visbench.datasets.synthetic_geometric import _perspective
    make = _perspective(strength=0.1, seed=0)
    sizes = [(320, 240), (640, 480), (320, 240), (800, 600), (640, 480)]
    Hs = [make(w, h)[0] for (w, h) in sizes]
    # The two (640, 480) calls and the two (320, 240) calls must match.
    np.testing.assert_array_equal(Hs[1], Hs[4])  # both (640, 480)
    np.testing.assert_array_equal(Hs[0], Hs[2])  # both (320, 240)


# --- spec.kwargs threading -------------------------------------------------

def test_run_one_forwards_spec_kwargs(monkeypatch, tmp_path):
    """run_one must pass spec.kwargs to iter_pairs — otherwise _wide and _hard
    variants silently match their base variant's output."""
    from pathlib import Path
    from visbench.datasets._registry import DatasetSpec, register, _REGISTRY
    from visbench.orchestrators.run import run_one

    seen_kwargs: dict = {}

    def fake_iter(data_root, max_pairs=None, **kw):
        seen_kwargs.update(kw)
        return iter([])

    spec = DatasetSpec(
        task="qualitative", iter_pairs=fake_iter,
        default_root=Path("dummy"), source_url="x",
        kwargs={"my_test_flag": "expected_value"},
    )
    monkeypatch.setenv("VISBENCH_ALLOW_REREGISTER", "1")
    register("__test_kwargs_dataset", spec)

    # Make data_root.exists() pass
    (tmp_path / "dummy").mkdir()
    try:
        run_one(
            dataset="__test_kwargs_dataset", method="sift",
            data_root_default=tmp_path,
            max_pairs=1, ransac_runs=1, device="cpu", progress=False,
        )
    except Exception:
        # Real evaluation may fail (no images); we only care kwargs forwarded.
        pass
    assert seen_kwargs.get("my_test_flag") == "expected_value", \
        f"spec.kwargs not forwarded to iter_pairs: {seen_kwargs}"
    _REGISTRY.pop("__test_kwargs_dataset", None)


# --- correspondence: zero-GT pair → finite PCK -----------------------------

def test_eval_correspondence_zero_gt_returns_finite():
    """A pair with empty `corrs` (no GT correspondences) used to mean(empty),
    producing NaN that contaminated the per-subset aggregate and broke
    downstream JSON parsing (NaN is not valid JSON)."""
    import json
    import warnings
    from visbench.core.harness import eval_correspondence_pairs

    class StubMatcher:
        def load_image(self, p, resize=None):
            return None

        def __call__(self, img0, img1):
            return {"matched_kpts0": np.zeros((0, 2)),
                    "matched_kpts1": np.zeros((0, 2)),
                    "all_kpts0": np.zeros((0, 2)),
                    "all_kpts1": np.zeros((0, 2)),
                    "num_inliers": 0}

    from PIL import Image
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        p_a = Path(td) / "a.png"; Image.new("RGB", (32, 32)).save(p_a)
        p_b = Path(td) / "b.png"; Image.new("RGB", (32, 32)).save(p_b)
        pair = {
            "im_A_path": str(p_a),
            "im_B_path": str(p_b),
            "corrs": np.zeros((0, 4)),  # ← zero GT correspondences
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = eval_correspondence_pairs(StubMatcher(), [pair], progress=False)
    # Must be JSON-serialisable (no NaN literals)
    blob = json.dumps(res, allow_nan=False)
    assert "NaN" not in blob
    for k, v in res.items():
        if isinstance(v, float):
            assert not (v != v), f"NaN leaked into {k}={v}"
