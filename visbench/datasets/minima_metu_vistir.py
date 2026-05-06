"""METU-VisTIR pose iterator (MINIMA / XoFTR protocol). Real RGB-thermal outdoor.

Source:  https://github.com/OnderT/XoFTR
License: Academic (per source repo).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.metrics.pose import compute_relative_pose
from ._registry import DatasetSpec, register


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if (data_root / "METU_VisTIR" / "index").exists() or (data_root / "index").exists():
        return data_root
    raise NotImplementedError(
        "METU-VisTIR: get the dataset from the XoFTR release\n"
        "(https://github.com/OnderT/XoFTR -- see the README 'Dataset' section) and\n"
        f"place under {data_root}/METU_VisTIR/ (must contain index/ and image dirs)."
    )


def _read_test_list(test_list: Path) -> list[str]:
    with open(test_list, "r") as f:
        return [ln.split()[0] for ln in f if ln.strip()]


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               split: str = "test") -> Iterator[dict]:
    data_root = Path(data_root)
    inner = data_root / "METU_VisTIR"
    if inner.exists():
        data_root = inner

    npz_dir = data_root / "index" / f"scene_info_{split}"
    test_list = data_root / "index" / "val_test_list" / f"{split}_list.txt"
    if not npz_dir.exists() or not test_list.exists():
        raise FileNotFoundError(f"METU-VisTIR layout missing under {data_root}")

    scene_names = _read_test_list(test_list)
    count = 0
    for name in scene_names:
        scene = np.load(npz_dir / name, allow_pickle=True)
        pair_infos = scene["pair_infos"]
        intrinsics = scene["intrinsics"]
        poses = scene["poses"]
        image_paths = scene["image_paths"]

        for id0, id1 in pair_infos:
            im0 = image_paths[id0][0]
            im1 = image_paths[id1][1]
            K0 = intrinsics[id0][0].astype(np.float64)
            K1 = intrinsics[id1][1].astype(np.float64)
            T0 = poses[id0]
            T1 = poses[id1]
            R, t = compute_relative_pose(T0[:3, :3], T0[:3, 3], T1[:3, :3], T1[:3, 3])
            yield {
                "im_A_path": str(data_root / im0),
                "im_B_path": str(data_root / im1),
                "K0": K0,
                "K1": K1,
                "R_0to1": R,
                "t_0to1": t,
                "scene": name,
                "subset": name.split("_scene")[0],
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


register("minima_metu_vistir", DatasetSpec(
    task="pose",
    iter_pairs=iter_pairs,
    default_root=Path("METU_VisTIR"),
    default_resize=640,
    download=download,
    source_url="https://github.com/OnderT/XoFTR",
    license="Academic",
))
