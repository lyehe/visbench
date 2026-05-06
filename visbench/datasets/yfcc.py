"""YFCC100M-4scene pose subset — SuperGlue/LoFTR legacy pose benchmark.

Source:  https://github.com/zju3dv/LoFTR (test-data tar)
License: YFCC100M is CC-BY (per-image varies); LoFTR test list is Apache 2.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ._registry import DatasetSpec, register


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if (data_root / "yfcc_test_pairs_with_gt.txt").exists():
        return data_root
    raise NotImplementedError(
        "YFCC: download yfcc_test_pairs_with_gt.txt + yfcc100m_test_pairs_images.tar from\n"
        "https://github.com/zju3dv/LoFTR (Reproduce the testing results) and extract under\n"
        f"{data_root}."
    )


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    pairs_file = data_root / "yfcc_test_pairs_with_gt.txt"
    if not pairs_file.exists():
        raise FileNotFoundError(f"YFCC pair file missing: {pairs_file}")
    count = 0
    for line in pairs_file.read_text().splitlines():
        tokens = line.strip().split()
        if len(tokens) < 2:
            continue
        imA_rel, imB_rel = tokens[0], tokens[1]
        floats = np.array([float(x) for x in tokens[2:]])
        K0 = floats[:9].reshape(3, 3)
        K1 = floats[9:18].reshape(3, 3)
        T = floats[18:34].reshape(4, 4)
        yield {
            "im_A_path": str(data_root / imA_rel),
            "im_B_path": str(data_root / imB_rel),
            "K0": K0, "K1": K1,
            "R_0to1": T[:3, :3],
            "t_0to1": T[:3, 3],
            "scene": Path(imA_rel).parts[0],
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


register("yfcc", DatasetSpec(
    task="pose", iter_pairs=iter_pairs, default_root=Path("yfcc"),
    default_resize=1200, download=download,
    source_url="https://github.com/zju3dv/LoFTR",
    license="YFCC100M CC-BY (varies per image)",
))
