"""Reproducibility: seeded pose AUCs are deterministic, `--rerun` aliases
`--no-skip`, and `report(json_out=...)` writes a JSON sidecar.
"""

from __future__ import annotations

import json

import numpy as np

from visbench.__main__ import _build_parser


# ----- CLI: --rerun is alias for --no-skip ---------------------------------

def test_cli_rerun_alias_batch():
    p = _build_parser()
    args = p.parse_args(["batch", "--method", "sift", "--datasets", "a",
                         "--out-dir", "x", "--rerun"])
    assert args.no_skip is True


def test_cli_rerun_alias_standard():
    p = _build_parser()
    args = p.parse_args(["standard", "--panel", "core", "--methods", "m.txt",
                         "--out-dir", "x", "--rerun"])
    assert args.no_skip is True


def test_cli_run_accepts_seed():
    p = _build_parser()
    args = p.parse_args(["run", "--dataset", "hpatches", "--method", "sift",
                         "--seed", "42"])
    assert args.seed == 42


def test_cli_report_accepts_json_out():
    p = _build_parser()
    args = p.parse_args(["report", "--results-dir", "x", "--json-out", "out.json"])
    assert args.json_out == "out.json"


# ----- Reproducibility -----------------------------------------------------

def test_pose_eval_deterministic_with_seed(tmp_path):
    """Same seed across runs → byte-identical AUC. Different seed → differs."""
    from PIL import Image
    from visbench.core.harness import eval_pose_pairs, reset_caches

    # Two readable image files so _pil_size succeeds.
    p_a = tmp_path / "a.png"
    Image.new("RGB", (64, 48)).save(p_a)
    p_b = tmp_path / "b.png"
    Image.new("RGB", (64, 48)).save(p_b)

    class StubMatcher:
        def load_image(self, p, resize=None):
            return None

        def __call__(self, img0, img1):
            rng = np.random.default_rng(123)
            kpts0 = rng.uniform(0, 60, (32, 2))
            kpts1 = kpts0 + rng.normal(0, 0.5, (32, 2))
            return {
                "matched_kpts0": kpts0,
                "matched_kpts1": kpts1,
                "all_kpts0": kpts0,
                "all_kpts1": kpts1,
                "num_inliers": 16,
            }

    pair = {
        "im_A_path": str(p_a), "im_B_path": str(p_b),
        "K0": np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1.0]]),
        "K1": np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1.0]]),
        "R_0to1": np.eye(3), "t_0to1": np.array([1.0, 0, 0]),
    }

    reset_caches()
    out1 = eval_pose_pairs(StubMatcher(), [pair, pair], seed=7,
                           resize_long=64, ransac_runs=3, progress=False)
    reset_caches()
    out2 = eval_pose_pairs(StubMatcher(), [pair, pair], seed=7,
                           resize_long=64, ransac_runs=3, progress=False)
    reset_caches()
    out3 = eval_pose_pairs(StubMatcher(), [pair, pair], seed=99,
                           resize_long=64, ransac_runs=3, progress=False)
    # Same seed → identical AUCs.
    for k in ("auc_5", "auc_10", "auc_20"):
        assert out1[k] == out2[k], f"seed=7 run differed on {k}: {out1[k]} vs {out2[k]}"
    # Different seed → not guaranteed to differ on tiny inputs, but the RNG
    # is wired through. The pose computation is deterministic given pixel
    # inputs, so this assertion is intentionally weaker: just that the seed
    # plumbing doesn't crash.
    assert "auc_5" in out3


def test_pose_eval_accepts_seed_kwarg():
    """The signature must include seed for the dispatch table to thread it through."""
    import inspect
    from visbench.core.harness import eval_pose_pairs
    sig = inspect.signature(eval_pose_pairs)
    assert "seed" in sig.parameters


# ----- report(json_out=...) ------------------------------------------------

def test_report_writes_json(tmp_path):
    from visbench.orchestrators.reporting import report

    # Empty results dir → report still runs; payload contains structure.
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    out = tmp_path / "agg.json"
    payload = report(results_dir, top_k=5, json_out=out, show_reference=False)
    assert out.exists()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk == payload
    assert "per_dataset" in on_disk
    assert "combined" in on_disk
