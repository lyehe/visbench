"""ETH3D high-res multi-view pose pairs.

Source:  https://www.eth3d.net/datasets
Paper:   Schöps et al., 'A Multi-View Stereo Benchmark with High-Resolution Images', CVPR 2017.
License: CC BY-NC-SA 4.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._registry import DatasetSpec, register


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if data_root.exists() and any(p.is_dir() and (p / "cameras.txt").exists() for p in data_root.iterdir()):
        return data_root
    raise NotImplementedError(
        "ETH3D: download the high-res multi-view (undistorted) splits from\n"
        f"https://www.eth3d.net/datasets and place <scene>/{{images,cameras.txt,images.txt,pairs.txt}}/ under {data_root}."
    )


def _parse_colmap_cameras(path: Path) -> dict:
    cams = {}
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        tok = line.split()
        cam_id = int(tok[0])
        fx, fy, cx, cy = float(tok[4]), float(tok[5]), float(tok[6]), float(tok[7])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        cams[cam_id] = K
    return cams


def _parse_colmap_images(path: Path) -> dict:
    out = {}
    lines = [l for l in path.read_text().splitlines() if l and not l.startswith("#")]
    i = 0
    while i < len(lines):
        tok = lines[i].split()
        qw, qx, qy, qz = map(float, tok[1:5])
        tx, ty, tz = map(float, tok[5:8])
        cam_id = int(tok[8])
        name = tok[9]
        q = np.array([qw, qx, qy, qz]); q /= np.linalg.norm(q) + 1e-12
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        t = np.array([tx, ty, tz])
        out[name] = (R, t, cam_id)
        i += 2
    return out


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    count = 0
    for scene in sorted(data_root.iterdir()):
        if not scene.is_dir():
            continue
        cam_file = scene / "cameras.txt"
        img_file = scene / "images.txt"
        pair_file = scene / "pairs.txt"
        if not (cam_file.exists() and img_file.exists() and pair_file.exists()):
            continue
        cams = _parse_colmap_cameras(cam_file)
        imgs = _parse_colmap_images(img_file)
        for line in pair_file.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            if a not in imgs or b not in imgs:
                continue
            R0, t0, cam0 = imgs[a]
            R1, t1, cam1 = imgs[b]
            R, t = compute_relative_pose(R0, t0, R1, t1)
            a_name = Path(a).name
            b_name = Path(b).name
            yield {
                "im_A_path": str(scene / "images" / a_name),
                "im_B_path": str(scene / "images" / b_name),
                "K0": cams[cam0], "K1": cams[cam1],
                "R_0to1": R, "t_0to1": t,
                "scene": scene.name,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


_KW = dict(
    download=download,
    source_url="https://www.eth3d.net/datasets",
    license="CC BY-NC-SA 4.0",
)
register("eth3d", DatasetSpec("pose", iter_pairs, Path("eth3d"), default_resize=1200, **_KW))
register("eth3d_highres", DatasetSpec("pose", iter_pairs, Path("eth3d_highres"), default_resize=1200, **_KW))
