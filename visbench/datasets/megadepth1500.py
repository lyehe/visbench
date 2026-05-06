"""MegaDepth-1500 pair iterator (LoFTR / RoMa / DKM protocol).

Source:  test split published with LoFTR — https://github.com/zju3dv/LoFTR
Paper:   Sun et al., 'LoFTR: Detector-Free Local Feature Matching with Transformers', CVPR 2021.
License: Apache 2.0 (LoFTR code); MegaDepth images: see https://www.cs.cornell.edu/projects/megadepth/

Layout::

    datasets/megadepth/
        0015_0.1_0.3.npz
        0015_0.3_0.5.npz
        0022_0.1_0.3.npz
        0022_0.3_0.5.npz
        0022_0.5_0.7.npz
        Undistorted_SfM/0015/...
        Undistorted_SfM/0022/...

The .npz files reference image paths relative to `data_root`. Run
`download(data_root)` to fetch the LoFTR-bundled tarball.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._registry import DatasetSpec, register
from ._download import download_url, extract_archive

# Public mirror used by LoFTR/ASpan/RoMa benchmark scripts.
_TARBALL_URL = "https://drive.google.com/uc?id=1Mx2WTL0ULePz9N5skOQXrYxuaG2W3hOE"

SCENE_FILES = (
    "0015_0.1_0.3.npz",
    "0015_0.3_0.5.npz",
    "0022_0.1_0.3.npz",
    "0022_0.3_0.5.npz",
    "0022_0.5_0.7.npz",
)


def download(data_root: Path | str) -> Path:
    """Print fetch instructions; full automation requires gdown of the LoFTR tar."""
    data_root = Path(data_root)
    if all((data_root / f).exists() for f in SCENE_FILES):
        return data_root
    raise NotImplementedError(
        "MegaDepth-1500 requires manual fetch. Download LoFTR's test split from\n"
        "  https://github.com/zju3dv/LoFTR (README -> Reproduce the testing results)\n"
        f"and extract scene .npz files + Undistorted_SfM/ into {data_root}."
    )


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               scene_files=SCENE_FILES) -> Iterator[dict]:
    data_root = Path(data_root)
    count = 0
    for scene_file in scene_files:
        npz_path = data_root / scene_file
        if not npz_path.exists():
            raise FileNotFoundError(f"MegaDepth scene file missing: {npz_path}")
        scene = np.load(npz_path, allow_pickle=True)
        pair_infos = scene["pair_infos"]
        intrinsics = scene["intrinsics"]
        poses = scene["poses"]
        im_paths = scene["image_paths"]
        for pi in pair_infos:
            idx1, idx2 = pi[0]
            K0 = intrinsics[idx1].copy()
            K1 = intrinsics[idx2].copy()
            T1 = poses[idx1]
            T2 = poses[idx2]
            R, t = compute_relative_pose(T1[:3, :3], T1[:3, 3], T2[:3, :3], T2[:3, 3])
            yield {
                "im_A_path": str(data_root / im_paths[idx1]),
                "im_B_path": str(data_root / im_paths[idx2]),
                "K0": K0,
                "K1": K1,
                "R_0to1": R,
                "t_0to1": t,
                "scene": scene_file,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


register("megadepth1500", DatasetSpec(
    task="pose",
    iter_pairs=iter_pairs,
    default_root=Path("megadepth"),
    default_resize=1200,
    download=download,
    source_url="https://github.com/zju3dv/LoFTR",
    license="Apache 2.0 (LoFTR); MegaDepth images per source",
))
