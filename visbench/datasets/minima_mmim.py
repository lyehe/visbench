"""Multi-Modality Image Matching DB iterator (Jiang 2021 / MINIMA protocol).

Source:  https://github.com/StaRainJ/Multi-modality-image-matching-database-metrics-methods
Paper:   Jiang et al., 'A Review of Multimodal Image Matching: Methods and Applications', 2021.
License: Academic (per source repo).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from ._registry import DatasetSpec, register

_TRANSLATION = np.array([[1, 0, -1], [0, 1, -1], [0, 0, 1]], dtype=np.float64)


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if (data_root / "Multimodal_Image_Matching_Datasets").exists():
        return data_root
    raise NotImplementedError(
        "MMIM: clone https://github.com/StaRainJ/Multi-modality-image-matching-database-metrics-methods\n"
        f"and place its `Multimodal_Image_Matching_Datasets/` under {data_root}."
    )


def _resolve_test_list(repo_root: Path, kind: str) -> Path:
    if kind == "medical":
        return repo_root / "test_list.txt"
    if kind == "remote_sensing":
        return repo_root / "test_list_2.txt"
    raise ValueError(f"Unknown subset_list '{kind}'")


def _read_subset_names(test_list: Path) -> list[str]:
    if not test_list.exists() or test_list.stat().st_size == 0:
        return []
    return [ln.strip() for ln in open(test_list, "r") if ln.strip()]


def _scan_subdirs_with_list(dataset_root: Path) -> list[str]:
    out: list[str] = []
    for list_json in sorted(dataset_root.rglob("list.json")):
        out.append(str(list_json.parent.relative_to(dataset_root)).replace("\\", "/") + "/")
    return out


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               subset_list: str = "all") -> Iterator[dict]:
    import scipy.io as sio
    data_root = Path(data_root)
    candidates = [
        data_root,
        data_root / "mmim_data",
        data_root / "mmim_data" / "Multi-modality-image-matching-database-metrics-methods",
    ]
    repo_root = next(
        (c for c in candidates if (c / "Multimodal_Image_Matching_Datasets").exists()),
        None,
    )
    if repo_root is None:
        raise FileNotFoundError(f"MMIM tree not found under any of: {candidates}")
    dataset_root = repo_root / "Multimodal_Image_Matching_Datasets"

    kinds = ["medical", "remote_sensing"] if subset_list == "all" else [subset_list]
    subset_names: list[str] = []
    for k in kinds:
        names = _read_subset_names(_resolve_test_list(repo_root, k))
        if not names:
            top = "Medical" if k == "medical" else "RemoteSensing"
            names = _scan_subdirs_with_list(dataset_root / top)
            names = [f"{top}/{s}" for s in names]
        subset_names.extend(names)

    count = 0
    for name in subset_names:
        sub_dir = dataset_root / name
        list_json = sub_dir / "list.json"
        if not list_json.exists():
            continue
        with open(list_json, "r") as f:
            data = json.load(f)
        for group, files in data.items():
            mat_path = sub_dir / files[0]
            img1 = sub_dir / files[1]
            img2 = sub_dir / files[2]
            if not (mat_path.exists() and img1.exists() and img2.exists()):
                continue
            mat = sio.loadmat(str(mat_path))
            T = np.asarray(mat["T"], dtype=np.float64).T
            T = T / T[2, 2]
            T = _TRANSLATION @ T @ np.linalg.inv(_TRANSLATION)
            T = T / T[2, 2]
            yield {
                "im_A_path": str(img2),
                "im_B_path": str(img1),
                "H_gt": T,
                "subset": name.rstrip("/"),
                "seq": group,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


register("minima_mmim", DatasetSpec(
    task="homography",
    iter_pairs=iter_pairs,
    default_root=Path("mmim"),
    default_resize=None,
    download=download,
    source_url="https://github.com/StaRainJ/Multi-modality-image-matching-database-metrics-methods",
    license="Academic",
))
