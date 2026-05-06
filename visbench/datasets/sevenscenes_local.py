"""7Scenes — frame-skip pairing variant (no pair-list required).

Source:  https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
Paper:   Shotton et al., CVPR 2013.
License: Microsoft Research License.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._registry import DatasetSpec, register

K_7SCENES = np.array([
    [525.0, 0, 320.0],
    [0, 525.0, 240.0],
    [0, 0, 1.0],
], dtype=np.float64)


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if any((data_root / s).exists() for s in ("chess", "fire", "office", "redkitchen")):
        return data_root
    raise NotImplementedError(
        "7Scenes: download per-scene zip archives from\n"
        "https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/<scene>.zip\n"
        f"and extract under {data_root}/<scene>/seq-NN/."
    )


def _load_pose(p: Path) -> np.ndarray:
    return np.array([[float(x) for x in line.split()]
                     for line in p.read_text().splitlines()], dtype=np.float64)


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               pair_skip: int = 30) -> Iterator[dict]:
    data_root = Path(data_root)
    count = 0
    for scene in sorted(data_root.iterdir()):
        if not scene.is_dir():
            continue
        for seq in sorted(scene.iterdir()):
            if not seq.is_dir() or not seq.name.startswith("seq"):
                continue
            frames = sorted(seq.glob("frame-*.color.png"))
            if len(frames) < 2:
                continue
            for i in range(0, len(frames) - pair_skip, pair_skip // 2 or 1):
                j = i + pair_skip
                if j >= len(frames):
                    break
                pose_i_wc = _load_pose(frames[i].with_suffix("").with_suffix(".pose.txt"))
                pose_j_wc = _load_pose(frames[j].with_suffix("").with_suffix(".pose.txt"))
                R_i_wc, t_i_wc = pose_i_wc[:3, :3], pose_i_wc[:3, 3]
                R_j_wc, t_j_wc = pose_j_wc[:3, :3], pose_j_wc[:3, 3]
                R_i, t_i = R_i_wc.T, -R_i_wc.T @ t_i_wc
                R_j, t_j = R_j_wc.T, -R_j_wc.T @ t_j_wc
                R, t = compute_relative_pose(R_i, t_i, R_j, t_j)
                yield {
                    "im_A_path": str(frames[i]), "im_B_path": str(frames[j]),
                    "K0": K_7SCENES, "K1": K_7SCENES,
                    "R_0to1": R, "t_0to1": t,
                    "scene": f"{scene.name}/{seq.name}",
                }
                count += 1
                if max_pairs is not None and count >= max_pairs:
                    return


_KW = dict(
    download=download,
    source_url="https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/",
    license="Microsoft Research License",
)

register("sevenscenes", DatasetSpec("pose", iter_pairs, Path("sevenscenes"),
                                    default_resize=640, kwargs={"pair_skip": 30}, **_KW))
register("sevenscenes_wide", DatasetSpec("pose", iter_pairs, Path("sevenscenes"),
                                         default_resize=640, kwargs={"pair_skip": 100}, **_KW))
