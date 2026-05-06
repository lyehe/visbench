"""TUM RGB-D SLAM — indoor RGB-D sequences with GT trajectories.

Source:  https://huggingface.co/datasets/voviktyl/TUM_RGBD-SLAM (mirror of TUM)
Paper:   Sturm et al., 'A Benchmark for the Evaluation of RGB-D SLAM Systems', IROS 2012.
License: CC BY 4.0 (TUM CVG).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._download import download_hf
from ._registry import DatasetSpec, register


_HF_REPO = "voviktyl/TUM_RGBD-SLAM"

# TUM RGB-D per-camera intrinsics (file_formats page).
K_FR1 = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1.0]], dtype=np.float64)
K_FR2 = np.array([[520.9, 0, 325.1], [0, 521.0, 249.7], [0, 0, 1.0]], dtype=np.float64)
K_FR3 = np.array([[535.4, 0, 320.1], [0, 539.2, 247.6], [0, 0, 1.0]], dtype=np.float64)


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if any(data_root.glob("rgbd_dataset_*/groundtruth.txt")):
        return data_root
    return download_hf(_HF_REPO, data_root, repo_type="dataset")


def _K_for_seq(seq_name: str) -> np.ndarray:
    if "freiburg2" in seq_name:
        return K_FR2
    if "freiburg3" in seq_name:
        return K_FR3
    return K_FR1


def _parse_groundtruth(p: Path) -> list[tuple]:
    rows = []
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        tok = line.split()
        ts = float(tok[0])
        tx, ty, tz = float(tok[1]), float(tok[2]), float(tok[3])
        qx, qy, qz, qw = float(tok[4]), float(tok[5]), float(tok[6]), float(tok[7])
        n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
        ])
        t = np.array([tx, ty, tz])
        rows.append((ts, R, t))
    rows.sort(key=lambda r: r[0])
    return rows


def _nearest(rows: list, ts: float):
    lo, hi = 0, len(rows) - 1
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if rows[mid][0] < ts:
            lo = mid
        else:
            hi = mid
    a, b = rows[lo], rows[hi]
    return a if abs(a[0] - ts) < abs(b[0] - ts) else b


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               pairing_lag: float = 2.0) -> Iterator[dict]:
    data_root = Path(data_root)
    count = 0
    for seq in sorted(data_root.iterdir()):
        if not seq.is_dir() or not seq.name.startswith("rgbd_dataset"):
            continue
        gt = seq / "groundtruth.txt"
        rgb_dir = seq / "rgb"
        if not (gt.exists() and rgb_dir.exists()):
            continue
        rows = _parse_groundtruth(gt)
        if not rows:
            continue
        frames = sorted(rgb_dir.glob("*.png"))
        if len(frames) < 2:
            continue
        ts_all = [float(f.stem) for f in frames]
        for i in range(0, len(frames), 10):
            t_i = ts_all[i]
            j = i
            while j < len(frames) and ts_all[j] - t_i < pairing_lag:
                j += 1
            if j >= len(frames):
                continue
            _, R0_wc, t0_wc = _nearest(rows, t_i)
            _, R1_wc, t1_wc = _nearest(rows, ts_all[j])
            R0, t0 = R0_wc.T, -R0_wc.T @ t0_wc
            R1, t1 = R1_wc.T, -R1_wc.T @ t1_wc
            R, t = compute_relative_pose(R0, t0, R1, t1)
            K = _K_for_seq(seq.name)
            yield {
                "im_A_path": str(frames[i]),
                "im_B_path": str(frames[j]),
                "K0": K,
                "K1": K,
                "R_0to1": R,
                "t_0to1": t,
                "scene": seq.name,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


_KW = dict(
    download=download,
    source_url=f"https://huggingface.co/datasets/{_HF_REPO}",
    license="CC BY 4.0",
)

register("tum_rgbd", DatasetSpec("pose", iter_pairs, Path("tum_rgbd"),
                                 default_resize=640, kwargs={"pairing_lag": 0.5}, **_KW))
register("tum_rgbd_wide", DatasetSpec("pose", iter_pairs, Path("tum_rgbd"),
                                      default_resize=640, kwargs={"pairing_lag": 2.0}, **_KW))
