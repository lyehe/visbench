"""CPU tests for HardMatch, SatAst, rotated variants, and the romaxx / LoMa method adapters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# --- H&Z symmetric epipolar distance matches the reference formula ----------

def _reference_sed(p1, p2, F, eps=1e-8):
    """hardmatch.benchmark._symmetrical_epipolar_distance (v1.0.0), copied verbatim in form."""
    ones = np.ones((*p1.shape[:-1], 1))
    a = np.concatenate([p1, ones], -1)
    b = np.concatenate([p2, ones], -1)
    l1 = a @ F.transpose(0, 2, 1)
    l2 = b @ F
    num = (b * l1).sum(-1) ** 2
    den = 1.0 / (np.linalg.norm(l1[..., :2], axis=-1) ** 2) + 1.0 / (np.linalg.norm(l2[..., :2], axis=-1) ** 2)
    return np.sqrt(num * den + eps)


def test_epi_distance_hz_matches_reference():
    from visbench.core.harness import _epi_distance_hz
    rng = np.random.default_rng(0)
    F = rng.normal(size=(3, 3))
    p0 = rng.uniform(0, 800, size=(50, 2))
    p1 = rng.uniform(0, 800, size=(50, 2))
    np.testing.assert_allclose(_epi_distance_hz(F, p0, p1), _reference_sed(p0[None], p1[None], F[None])[0], rtol=1e-9)


# --- rotation helpers ---------------------------------------------------------

def test_rotate_camera_matches_rotated_projection():
    from visbench.datasets._rotation import rotate_camera, rotate_points
    rng = np.random.default_rng(1)
    W = 640
    K = np.array([[500.0, 0, 330.0], [0, 510.0, 240.0], [0, 0, 1]])
    ang = 0.2
    R = np.array([[np.cos(ang), 0, np.sin(ang)], [0, 1, 0], [-np.sin(ang), 0, np.cos(ang)]])
    t = np.array([0.3, -0.1, 0.2])
    X = rng.uniform([-2, -2, 4], [2, 2, 8], size=(30, 3))   # points in camera-0 frame
    Xb = X @ R.T + t
    xb = (Xb @ K.T)
    xb = xb[:, :2] / xb[:, 2:]
    K_rot, R_rot, t_rot = rotate_camera(K, R, t, W)
    Xr = X @ R_rot.T + t_rot
    xr = Xr @ K_rot.T
    xr = xr[:, :2] / xr[:, 2:]
    np.testing.assert_allclose(xr, rotate_points(xb, W), atol=1e-9)


def test_rotated_image_matches_point_map(tmp_path):
    from visbench.datasets._rotation import rotate_points, rotated_image
    arr = np.zeros((40, 60, 3), dtype=np.uint8)
    arr[10, 50] = 255                                     # pixel centre (50.5, 10.5)
    src = tmp_path / "b.png"
    Image.fromarray(arr).save(src)
    out, W, _ = rotated_image(src, tmp_path / "cache", Path("b.png"))
    rot = np.asarray(Image.open(out))
    y, x = np.argwhere(rot[..., 0] == 255)[0]
    np.testing.assert_allclose(rotate_points(np.array([[50.5, 10.5]]), W)[0], [x + 0.5, y + 0.5])


# --- HardMatch loader ---------------------------------------------------------

def _make_hardmatch(root: Path, n_cat=100, corr_rows=10):
    for c in range(n_cat):
        d = root / f"cat{c:03d}" / "obj" / "pair_0001"
        d.mkdir(parents=True)
        Image.new("RGB", (32, 24)).save(d / "A_lr.jpg")
        Image.new("RGB", (32, 24)).save(d / "B_lr.jpg")
        rows = corr_rows if c != 5 else 4                 # one pair with too few annotations
        txt = "\n".join("1 2 3 4" for _ in range(rows))
        (d / "corresp.txt").write_text(txt)
        (d / "corresp_lr.txt").write_text(txt)
        (d / "categories.txt").write_text("dynamic\n" if c == 7 else "outdoor\n")


def test_hardmatch_split_and_filters(tmp_path):
    from visbench.datasets.correspondence.hardmatch import iter_pairs, split_pairs
    _make_hardmatch(tmp_path)
    test, val = split_pairs(tmp_path, "test"), split_pairs(tmp_path, "val")
    assert len(test) == 90 and len(val) == 10 and not set(test) & set(val)
    yielded = {p["scene"] for p in iter_pairs(tmp_path, split="test")} | \
              {p["scene"] for p in iter_pairs(tmp_path, split="val")}
    assert "cat005" not in yielded and "cat007" not in yielded and len(yielded) == 98


def test_hardmatch_requires_100_categories(tmp_path):
    from visbench.datasets.correspondence.hardmatch import iter_pairs
    _make_hardmatch(tmp_path, n_cat=99)
    with pytest.raises(ValueError):
        list(iter_pairs(tmp_path))


# --- SatAst task --------------------------------------------------------------

def test_point_homography_perfect_matches(tmp_path):
    from visbench.core.harness import eval_point_homography_pairs
    Image.new("RGB", (200, 100)).save(tmp_path / "a.png")
    Image.new("RGB", (200, 100)).save(tmp_path / "b.png")
    H = np.array([[0.9, 0.05, 12.0], [-0.04, 1.1, 5.0], [1e-4, 0.0, 1.0]])

    def warp(p):
        q = np.c_[p, np.ones(len(p))] @ H.T
        return q[:, :2] / q[:, 2:]

    grid = np.stack(np.meshgrid(np.linspace(5, 150, 12), np.linspace(5, 80, 8)), -1).reshape(-1, 2)

    class Stub:
        def load_image(self, p, resize=None):
            return str(p)

        def __call__(self, a, b):
            return {"matched_kpts0": grid, "matched_kpts1": warp(grid)}

    pts = np.array([[20.0, 30.0], [100.0, 50.0], [150.0, 70.0]])
    pair = {"im_A_path": str(tmp_path / "a.png"), "im_B_path": str(tmp_path / "b.png"),
            "pts_A": pts, "pts_B": warp(pts)}
    out = eval_point_homography_pairs(Stub(), [pair], progress=False)
    assert out["auc_10px"] > 0.95 and out["num_points"] == 3


def test_satast_loader_reads_annotations(tmp_path):
    from visbench.datasets.homography.satast import iter_pairs
    (tmp_path / "annotations-v1").mkdir()
    (tmp_path / "data" / "X").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(tmp_path / "data" / "X" / "q.jpg")
    Image.new("RGB", (8, 8)).save(tmp_path / "data" / "X" / "p.jpg")
    ann = {"query_path": "data\\X\\q.jpg", "pred_path": "data\\X\\p.jpg",
           "correspondences": [{"pts_src": [[0, 0]], "pts_dst": [[9, 9]]},
                               {"pts_src": [[1, 2], [3, 4]], "pts_dst": [[5, 6], [7, 8]]}]}
    (tmp_path / "annotations-v1" / "annotation_X.json").write_text(json.dumps(ann))
    (p,) = list(iter_pairs(tmp_path))
    assert p["pts_A"].tolist() == [[1, 2], [3, 4]] and p["scene"] == "X"


# --- method adapters ----------------------------------------------------------

def test_romaxx_adapter_dense(tmp_path):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "inference_mode"):
        pytest.skip("real torch not installed (conftest stub)")
    from visbench.methods.romaxx_loader import RomaxxMatcher, parse

    assert parse("romaxx:flexglue-trt:C:/bundles/core") == ("flexglue-trt", "C:/bundles/core")

    class FakeDense:
        def match(self, a, b):
            return {}

        def sample(self, preds, n):
            return torch.tensor([[0.0, 0.0, 0.5, 0.5]]), torch.ones(1)

        def to_pixel_coordinates(self, m, H_A, W_A, H_B, W_B):
            return (torch.tensor([[W_A / 2, H_A / 2]]), torch.tensor([[W_B * 0.75, H_B * 0.75]]))

    m = RomaxxMatcher("romatrt", None, model=FakeDense())
    out = m(torch.zeros(3, 40, 60), torch.zeros(3, 20, 80))
    np.testing.assert_allclose(out["matched_kpts0"], [[30, 20]])
    np.testing.assert_allclose(out["matched_kpts1"], [[60, 15]])
    Image.new("RGB", (60, 40)).save(tmp_path / "i.png")
    assert tuple(m.load_image(tmp_path / "i.png", resize=(20, 30)).shape) == (3, 20, 30)


def test_romaxx_method_resolves_without_romaxx_installed():
    from visbench.methods import get
    spec = get("romaxx:lomatrt:loma_G")
    assert spec.kind == "romaxx" and callable(spec.builder)


def test_vismatch_loma_aliases(monkeypatch):
    import sys
    import types
    calls = {}
    fake = types.SimpleNamespace(get_matcher=lambda name, device, **kw: calls.update(name=name, **kw) or "m")
    monkeypatch.setitem(sys.modules, "vismatch", fake)
    from visbench.methods.vismatch_loader import build_vismatch
    assert build_vismatch("loma-g", "cpu") == "m"
    assert calls == {"name": "loma", "arch": "LoMa-G", "max_num_keypoints": 4096}


def test_romaxx_adapter_lomatrt_maps_back_from_engine_size(monkeypatch):
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "inference_mode"):
        pytest.skip("real torch not installed (conftest stub)")
    from visbench.methods.romaxx_loader import RomaxxMatcher

    seen = {}

    class FakeLoMa:
        def match_points(self, a, b):
            seen["shapes"] = (tuple(a.shape), tuple(b.shape))
            return np.array([[256.0, 128.0]]), np.array([[512.0, 0.0]])

    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: self)
    m = RomaxxMatcher("lomatrt", None, model=FakeLoMa())
    out = m(torch.zeros(3, 100, 200), torch.zeros(3, 300, 50))
    assert seen["shapes"] == ((1, 3, 512, 512), (1, 3, 512, 512))
    np.testing.assert_allclose(out["matched_kpts0"], [[100.0, 25.0]])
    np.testing.assert_allclose(out["matched_kpts1"], [[50.0, 0.0]])
