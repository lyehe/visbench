"""7Scenes — Microsoft RGB-D indoor relocalization (Kinect v1) — pair-list variant.

Source:  https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
Paper:   Shotton et al., 'Scene Coordinate Regression Forests for Camera Relocalization in RGB-D Images', CVPR 2013.
License: Microsoft Research License.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._registry import DatasetSpec, register

K_7SCENES = np.array([
    [585.0, 0.0, 320.0],
    [0.0, 585.0, 240.0],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if any((data_root / s).exists() for s in ("chess", "fire", "office", "redkitchen")):
        return data_root
    raise NotImplementedError(
        "7Scenes: download per-scene 7z archives from the Microsoft Research page\n"
        "(https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)\n"
        f"and extract under {data_root}/<scene>/seq-XX/. Add a pairs.txt per scene."
    )


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    count = 0
    for scene in sorted(data_root.iterdir()):
        if not scene.is_dir():
            continue
        pair_file = scene / "pairs.txt"
        if not pair_file.exists():
            continue
        for line in pair_file.read_text().splitlines():
            tok = line.strip().split()
            if len(tok) < 2:
                continue
            a_stem, b_stem = tok[0], tok[1]
            a_color = data_root / f"{a_stem}.color.png"
            b_color = data_root / f"{b_stem}.color.png"
            a_pose = data_root / f"{a_stem}.pose.txt"
            b_pose = data_root / f"{b_stem}.pose.txt"
            if not all(p.exists() for p in (a_color, b_color, a_pose, b_pose)):
                continue
            Tw0 = np.loadtxt(a_pose); Tw1 = np.loadtxt(b_pose)
            T0 = np.linalg.inv(Tw0); T1 = np.linalg.inv(Tw1)
            R, t = compute_relative_pose(T0[:3, :3], T0[:3, 3], T1[:3, :3], T1[:3, 3])
            yield {
                "im_A_path": str(a_color), "im_B_path": str(b_color),
                "K0": K_7SCENES, "K1": K_7SCENES,
                "R_0to1": R, "t_0to1": t,
                "scene": scene.name,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


register("seven_scenes", DatasetSpec(
    task="pose", iter_pairs=iter_pairs, default_root=Path("seven_scenes"),
    default_resize=640, download=download,
    source_url="https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/",
    license="Microsoft Research License",
))
