"""Panels name only registered datasets / well-formed methods; `visbench check` on fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PANELS = Path(__file__).resolve().parent.parent / "panels"


def _names(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


@pytest.mark.parametrize("panel", sorted(p.name for p in PANELS.glob("*.txt") if "methods" not in p.name))
def test_dataset_panels_are_registered(panel):
    from visbench.datasets import available
    missing = [n for n in _names(PANELS / panel) if n not in set(available())]
    assert not missing, f"{panel}: unregistered datasets {missing}"


@pytest.mark.parametrize("panel", sorted(p.name for p in PANELS.glob("*methods.txt")))
def test_method_panels_resolve(panel):
    from visbench.methods import get
    for name in _names(PANELS / panel):
        assert callable(get(name).builder), name


def test_check_flags_out_of_image_annotations(tmp_path, monkeypatch, capsys):
    from visbench.datasets._registry import _REGISTRY, DatasetSpec, register
    from visbench.orchestrators.check import check_dataset

    for i in range(4):
        Image.new("RGB", (20, 10)).save(tmp_path / f"a{i}.png")
    bad = np.array([[1.0, 1.0, 50.0, 5.0]])           # x=50 outside a 20-wide image

    def it(root, max_pairs=None):
        for i in range(4):
            yield {"im_A_path": str(tmp_path / f"a{i}.png"), "im_B_path": str(tmp_path / f"a{i}.png"),
                   "corrs": bad if i < 2 else np.array([[1.0, 1.0, 2.0, 2.0]]), "scene": str(i)}

    monkeypatch.setenv("VISBENCH_ALLOW_REREGISTER", "1")
    register("__check_corr", DatasetSpec("correspondence", it, Path("x"), source_url="x"))
    try:
        assert not check_dataset("__check_corr", tmp_path, n_pairs=4)
        assert "2 with problems" in capsys.readouterr().out
    finally:
        _REGISTRY.pop("__check_corr", None)
