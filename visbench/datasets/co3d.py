"""CO3Dv2 — Common Objects in 3D (Facebook/Meta).

Source:  https://github.com/facebookresearch/co3d
Paper:   Reizenstein et al., 'Common Objects in 3D', ICCV 2021.
License: CC BY-NC 4.0.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator

import numpy as np

from ._registry import DatasetSpec, register


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if data_root.exists() and any(p.is_dir() for p in data_root.iterdir()):
        return data_root
    raise NotImplementedError(
        "CO3Dv2: clone https://github.com/facebookresearch/co3d and run\n"
        f"`download.sh` (or use the provided Dockerfile). Place categories under {data_root}."
    )


def _load_frame_annotations(path: Path) -> dict:
    if path.suffix == ".jgz":
        with gzip.open(path, "rt") as f:
            data = json.load(f)
    else:
        data = json.loads(path.read_text())
    out = {}
    for f in data:
        name = Path(f["image"]["path"]).stem
        out[name] = f["viewpoint"]
    return out


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    count = 0
    for cat in sorted(data_root.iterdir()):
        if not cat.is_dir():
            continue
        for seq in sorted(cat.iterdir()):
            if not seq.is_dir():
                continue
            pair_file = seq / "pairs.txt"
            ann_file = None
            for cand in ("frame_annotations.jgz", "frame_annotations.json"):
                if (seq / cand).exists():
                    ann_file = seq / cand
                    break
            if not (pair_file.exists() and ann_file):
                continue
            ann = _load_frame_annotations(ann_file)
            for line in pair_file.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                a, b = parts[0], parts[1]
                if a not in ann or b not in ann:
                    continue
                va, vb = ann[a], ann[b]
                R0 = np.array(va["R"]).reshape(3, 3); t0 = np.array(va["T"]).reshape(3)
                R1 = np.array(vb["R"]).reshape(3, 3); t1 = np.array(vb["T"]).reshape(3)
                fx, fy = va["focal_length"]; cx, cy = va["principal_point"]
                K0 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                fx, fy = vb["focal_length"]; cx, cy = vb["principal_point"]
                K1 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                R = R1 @ R0.T
                t = -R @ t0 + t1
                yield {
                    "im_A_path": str(seq / "images" / f"{a}.jpg"),
                    "im_B_path": str(seq / "images" / f"{b}.jpg"),
                    "K0": K0, "K1": K1,
                    "R_0to1": R, "t_0to1": t,
                    "scene": f"{cat.name}/{seq.name}",
                }
                count += 1
                if max_pairs is not None and count >= max_pairs:
                    return


register("co3d", DatasetSpec(
    task="pose", iter_pairs=iter_pairs, default_root=Path("co3d"),
    default_resize=800, download=download,
    source_url="https://github.com/facebookresearch/co3d",
    license="CC BY-NC 4.0",
))
