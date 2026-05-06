"""MegaDepth-1500-Syn pose iterator (MINIMA cross-modal).

Reuses MegaDepth-1500's NPZs but swaps im_B for the synthesized cross-modal
variant from `Megadepth-1500-syn/<Modality>`. Six modalities are exposed as
separate dataset names: depth, event, infrared, normal, paint, sketch.

Source:  https://github.com/LSXI7/MINIMA (see Megadepth-1500-syn release)
License: inherits MegaDepth + MINIMA.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._registry import DatasetSpec, register
from .megadepth1500 import SCENE_FILES

MODALITY_DIRS = {
    "depth":    "Depth",
    "event":    "Event",
    "infrared": "Infrared",
    "normal":   "Normal",
    "paint":    "Paint",
    "sketch":   "Sketch",
}


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if (data_root / "test" / "Megadepth-1500-syn").exists():
        return data_root
    raise NotImplementedError(
        "MD1500-Syn: download from MINIMA release and extract\n"
        f"  test/Megadepth-1500-syn/<Depth,Event,Infrared,Normal,Paint,Sketch>/ under {data_root}.\n"
        "Also requires megadepth1500's NPZs in a sibling `megadepth/` dir."
    )


def _make_iter(modality: str):
    syn_subdir = MODALITY_DIRS[modality]

    def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
        data_root = Path(data_root)
        candidates = [
            data_root / "test" / "Megadepth-1500-syn",
            data_root / "megadepth_syn" / "test" / "Megadepth-1500-syn",
            data_root,
        ]
        syn_root = next((c for c in candidates if (c / syn_subdir).exists()), None)
        if syn_root is None:
            raise FileNotFoundError(f"Megadepth-1500-syn/{syn_subdir}/ not found under {candidates}")
        syn_modality_root = syn_root / syn_subdir

        md_root = data_root.parent / "megadepth" if (data_root.parent / "megadepth").exists() else data_root.parent
        npz_dir_candidates = [
            md_root,
            md_root / "scene_info",
            md_root / "index" / "scene_info_0.1_0.7_no_sfm",
        ]
        npz_dir = next((c for c in npz_dir_candidates if all((c / s).exists() for s in SCENE_FILES)), None)
        if npz_dir is None:
            raise FileNotFoundError(f"MD1500 scene NPZs not found under {npz_dir_candidates}")

        count = 0
        for scene_file in SCENE_FILES:
            scene = np.load(npz_dir / scene_file, allow_pickle=True)
            pair_infos = scene["pair_infos"]
            intrinsics = scene["intrinsics"]
            poses = scene["poses"]
            im_paths = scene["image_paths"]

            for pi in pair_infos:
                idx1, idx2 = pi[0]
                K0 = intrinsics[idx1].copy()
                K1 = intrinsics[idx2].copy()
                T1 = poses[idx1]; T2 = poses[idx2]
                R, t = compute_relative_pose(T1[:3, :3], T1[:3, 3], T2[:3, :3], T2[:3, 3])
                im_A_path = md_root / im_paths[idx1]
                im_B_rel = im_paths[idx2]
                im_B_path = (syn_modality_root / im_B_rel).with_suffix(".png")
                yield {
                    "im_A_path": str(im_A_path),
                    "im_B_path": str(im_B_path),
                    "K0": K0, "K1": K1,
                    "R_0to1": R, "t_0to1": t,
                    "scene": scene_file,
                    "subset": modality,
                }
                count += 1
                if max_pairs is not None and count >= max_pairs:
                    return

    return iter_pairs


for mod in MODALITY_DIRS:
    register(f"minima_md1500_{mod}", DatasetSpec(
        task="pose",
        iter_pairs=_make_iter(mod),
        default_root=Path("megadepth_syn"),
        default_resize=1200,
        download=download,
        source_url="https://github.com/LSXI7/MINIMA",
        license="Inherits MegaDepth + MINIMA",
    ))
