"""BlendedMVS — multi-view stereo dataset with per-view poses.

Source:  https://huggingface.co/datasets/infinity1096/blendedmvs_processed
Paper:   Yao et al., 'BlendedMVS: A Large-scale Dataset for Generalized Multi-view Stereo Networks', CVPR 2020.
License: CC BY 4.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._download import download_hf
from ._registry import DatasetSpec, register

_HF_REPO = "infinity1096/blendedmvs_processed"


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if any(data_root.glob("*/*.jpg")):
        return data_root
    return download_hf(_HF_REPO, data_root, repo_type="dataset")


def _load_pose(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    K = d["intrinsics"].astype(np.float64)
    R_cw = d["R_cam2world"].astype(np.float64)
    t_cw = d["t_cam2world"].astype(np.float64)
    return K, R_cw, t_cw


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               pair_skip: int = 3) -> Iterator[dict]:
    data_root = Path(data_root)
    count = 0
    for scene in sorted(data_root.iterdir()):
        if not scene.is_dir():
            continue
        jpgs = sorted(scene.glob("*.jpg"))
        for i in range(0, len(jpgs) - pair_skip, pair_skip // 2 or 1):
            j = i + pair_skip
            if j >= len(jpgs):
                break
            npz_a = jpgs[i].with_suffix(".npz")
            npz_b = jpgs[j].with_suffix(".npz")
            if not (npz_a.exists() and npz_b.exists()):
                continue
            K0, R_A_wc, t_A_wc = _load_pose(npz_a)
            K1, R_B_wc, t_B_wc = _load_pose(npz_b)
            R_A, t_A = R_A_wc.T, -R_A_wc.T @ t_A_wc
            R_B, t_B = R_B_wc.T, -R_B_wc.T @ t_B_wc
            R, t = compute_relative_pose(R_A, t_A, R_B, t_B)
            yield {
                "im_A_path": str(jpgs[i]),
                "im_B_path": str(jpgs[j]),
                "K0": K0,
                "K1": K1,
                "R_0to1": R,
                "t_0to1": t,
                "scene": scene.name,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


_KW = dict(
    download=download,
    source_url=f"https://huggingface.co/datasets/{_HF_REPO}",
    license="CC BY 4.0",
)

register("blendedmvs", DatasetSpec("pose", iter_pairs, Path("blendedmvs"),
                                   default_resize=768, kwargs={"pair_skip": 1}, **_KW))
register("blendedmvs_wide", DatasetSpec("pose", iter_pairs, Path("blendedmvs"),
                                        default_resize=768, kwargs={"pair_skip": 3}, **_KW))
