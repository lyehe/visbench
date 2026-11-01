"""Regression tests for pair sampling, result reuse, per-task caps and reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from visbench.core.sampling import sample_pairs


def _pairs(scenes: dict[str, int]):
    for scene, n in scenes.items():
        for i in range(n):
            yield {"scene": scene, "i": i}


def test_sample_pairs_no_cap_keeps_order():
    pairs = list(_pairs({"a": 3, "b": 2}))
    assert list(sample_pairs(iter(pairs), None)) == pairs


def test_sample_pairs_cap_covers_every_scene():
    out = list(sample_pairs(_pairs({f"s{k}": 50 for k in range(12)}), 24, seed=0))
    assert len(out) == 24
    assert {p["scene"] for p in out} == {f"s{k}" for k in range(12)}


def test_sample_pairs_is_seeded():
    a = list(sample_pairs(_pairs({"x": 40, "y": 40}), 10, seed=3))
    b = list(sample_pairs(_pairs({"x": 40, "y": 40}), 10, seed=3))
    c = list(sample_pairs(_pairs({"x": 40, "y": 40}), 10, seed=4))
    assert a == b and a != c


def test_sample_pairs_cap_above_total_returns_all():
    assert len(list(sample_pairs(_pairs({"a": 3, "b": 2}), 100))) == 5


def _record(tmp_path: Path, **overrides) -> Path:
    from visbench.orchestrators.run import RESULT_SCHEMA_VERSION
    rec = {"schema_version": RESULT_SCHEMA_VERSION, "dataset": "d", "method": "m",
           "max_pairs": 100, "seed": 0, "metrics": {"num_pairs": 100, "auc_5": 0.5}}
    rec.update(overrides)
    path = tmp_path / "d__m.json"
    path.write_text(json.dumps(rec), encoding="utf-8")
    return path


def test_existing_result_reused_only_when_current(tmp_path):
    from visbench.orchestrators.batch import existing_result_is_current
    assert existing_result_is_current(_record(tmp_path), 100, 0)
    assert not existing_result_is_current(_record(tmp_path), 1500, 0)
    assert not existing_result_is_current(_record(tmp_path), 100, 1)
    assert not existing_result_is_current(_record(tmp_path, schema_version=None), 100, 0)
    assert not existing_result_is_current(_record(tmp_path, metrics={"num_pairs": 0}), 100, 0)


def test_per_task_caps_are_independent():
    from visbench.orchestrators.batch import per_dataset_max_pairs
    caps = per_dataset_max_pairs(["megadepth1500", "hpatches"], pose=1500, homography=100)
    assert caps == {"megadepth1500": 1500, "hpatches": 100}


def test_parallel_standard_forwards_both_caps(tmp_path, monkeypatch):
    from visbench.orchestrators import standard
    panel = tmp_path / "p.txt"
    panel.write_text("megadepth1500\nhpatches\n")
    methods = tmp_path / "m.txt"
    methods.write_text("sift\n")
    seen = []

    class FakePopen:
        returncode = 0

        def __init__(self, argv, **kw):
            seen.append(argv)

        def communicate(self, timeout=None):
            return "", ""

    monkeypatch.setattr(standard.subprocess, "Popen", FakePopen)
    standard.run_standard_parallel(panel, methods, tmp_path / "out", tmp_path,
                                   max_pairs_pose=1500, max_pairs_homography=100)
    argv = seen[0]
    assert argv[argv.index("--max-pairs-pose") + 1] == "1500"
    assert argv[argv.index("--max-pairs-homography") + 1] == "100"
    assert "--max-pairs" not in argv


def test_report_uses_task_metrics_and_na(tmp_path, capsys):
    from visbench.orchestrators.reporting import report
    rd = tmp_path / "r"
    rd.mkdir()
    (rd / "hpatches__a.json").write_text(json.dumps(
        {"dataset": "hpatches", "method": "a", "metrics": {"all_auc_3": 0.7, "all_auc_5": 0.8}}))
    (rd / "megadepth1500__a.json").write_text(json.dumps(
        {"dataset": "megadepth1500", "method": "a", "metrics": {"auc_5": 0.5}}))
    payload = report(rd, panel=["hpatches", "megadepth1500"], show_reference=False)
    assert payload["per_dataset"]["hpatches"]["task"] == "homography"
    assert payload["per_dataset"]["hpatches"]["rows"][0]["values"] == [0.7, 0.8, None]
    assert payload["per_dataset"]["megadepth1500"]["rows"][0]["values"] == [0.5, None, None]
    assert payload["matrix"]["a"] == {"hpatches": 0.7, "megadepth1500": 0.5}
    assert "n/a" in capsys.readouterr().out


def test_eth3d_images_txt_keeps_blank_points_lines(tmp_path):
    from visbench.datasets.pose.eth3d import _parse_colmap_images
    path = tmp_path / "images.txt"
    path.write_text(
        "# header\n"
        "1 1 0 0 0 0 0 0 1 a.jpg\n"
        "\n"
        "2 1 0 0 0 1 2 3 1 b.jpg\n"
        "10 20 7 11 21 8\n"
    )
    imgs = _parse_colmap_images(path)
    assert imgs["a.jpg"][3] == set()
    assert list(imgs["b.jpg"][1]) == [1.0, 2.0, 3.0]
    assert imgs["b.jpg"][3] == {7, 8}


def test_run_one_raises_on_empty_dataset(monkeypatch, tmp_path):
    from visbench.core.sampling import EmptyPairsError
    from visbench.datasets._registry import _REGISTRY, DatasetSpec, register
    from visbench.orchestrators.run import run_one

    monkeypatch.setenv("VISBENCH_ALLOW_REREGISTER", "1")
    register("__empty_pose", DatasetSpec(task="pose", iter_pairs=lambda root, max_pairs=None: iter([]),
                                         default_root=Path("empty"), source_url="x"))
    (tmp_path / "empty").mkdir()
    try:
        with pytest.raises(EmptyPairsError):
            run_one(dataset="__empty_pose", method="sift", data_root_default=tmp_path,
                    device="cpu", progress=False, out=tmp_path / "o.json")
        assert not (tmp_path / "o.json").exists()
    finally:
        _REGISTRY.pop("__empty_pose", None)
