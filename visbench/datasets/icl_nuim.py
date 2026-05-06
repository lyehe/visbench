"""ICL-NUIM — synthetic indoor RGB-D with GT trajectories.

Source:  https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html
Paper:   Handa et al., 'A Benchmark for RGB-D Visual Odometry, 3D Reconstruction and SLAM', ICRA 2014.
License: CC BY-NC.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._registry import DatasetSpec, register

K_ICL = np.array([
    [481.20, 0, 319.50],
    [0, -480.00, 239.50],
    [0, 0, 1.0],
], dtype=np.float64)


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if any(data_root.glob("*.gt.freiburg")):
        return data_root
    raise NotImplementedError(
        "ICL-NUIM: download a sequence (e.g. 'living-room') from\n"
        f"https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html and place under {data_root}."
    )


def _parse_gt_freiburg(p: Path) -> list[tuple[int, np.ndarray, np.ndarray]]:
    out = []
    for line in p.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        tok = line.split()
        fid = int(tok[0])
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
        out.append((fid, R, t))
    out.sort(key=lambda r: r[0])
    return out


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               pair_skip: int = 30) -> Iterator[dict]:
    data_root = Path(data_root)
    gt_files = list(data_root.glob("*.gt.freiburg"))
    if not gt_files:
        return
    gt = _parse_gt_freiburg(gt_files[0])
    rgb_dir = data_root / "rgb"
    count = 0
    for i in range(0, len(gt) - pair_skip, pair_skip // 2 or 1):
        j = i + pair_skip
        if j >= len(gt):
            break
        fid_a, R_a_wc, t_a_wc = gt[i]
        fid_b, R_b_wc, t_b_wc = gt[j]
        im_a = rgb_dir / f"{fid_a}.png"
        im_b = rgb_dir / f"{fid_b}.png"
        if not (im_a.exists() and im_b.exists()):
            continue
        R_a, t_a = R_a_wc.T, -R_a_wc.T @ t_a_wc
        R_b, t_b = R_b_wc.T, -R_b_wc.T @ t_b_wc
        R, t = compute_relative_pose(R_a, t_a, R_b, t_b)
        yield {
            "im_A_path": str(im_a), "im_B_path": str(im_b),
            "K0": K_ICL, "K1": K_ICL,
            "R_0to1": R, "t_0to1": t,
            "scene": "icl_nuim_living_room",
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


_KW = dict(
    download=download,
    source_url="https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html",
    license="CC BY-NC",
)

register("icl_nuim", DatasetSpec("pose", iter_pairs, Path("icl_nuim"),
                                 default_resize=640, kwargs={"pair_skip": 30}, **_KW))
register("icl_nuim_wide", DatasetSpec("pose", iter_pairs, Path("icl_nuim"),
                                      default_resize=640, kwargs={"pair_skip": 100}, **_KW))
