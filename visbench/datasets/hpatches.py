"""HPatches sequences (illumination + viewpoint).

Source:  https://github.com/hpatches/hpatches-dataset
Paper:   Balntas et al., 'HPatches: A benchmark and evaluation of handcrafted
         and learned local descriptors', CVPR 2017.
License: CC BY-NC-SA 4.0 — non-commercial research only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ._registry import DatasetSpec, register
from ._download import download_hf, extract_archive

_HF_REPO = "vbalnt/hpatches"
_HF_ARCHIVE = "hpatches-sequences-release.zip"

DEGENERATE = frozenset({
    "i_contruction", "i_crownnight", "i_dc", "i_pencils", "i_whitebuilding",
    "v_artisans", "v_astronautis", "v_talent",
})


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    target = data_root / "hpatches-sequences-release"
    if target.exists() and any(target.iterdir()):
        return data_root
    data_root.mkdir(parents=True, exist_ok=True)
    # The official icvl.ee.ic.ac.uk host is dead; use the author's HF mirror.
    download_hf(_HF_REPO, data_root, repo_type="dataset",
                allow_patterns=[_HF_ARCHIVE])
    archive = data_root / _HF_ARCHIVE
    if archive.exists() and not target.exists():
        extract_archive(archive, data_root)
        try:
            archive.unlink()
        except OSError:
            pass
    return data_root


def iter_pairs(data_root: Path | str, max_pairs: int | None = None,
               skip_degenerate: bool = True) -> Iterator[dict]:
    data_root = Path(data_root)
    seq_root = data_root / "hpatches-sequences-release"
    if not seq_root.exists():
        seq_root = data_root
    count = 0
    for seq in sorted(seq_root.iterdir()):
        if not seq.is_dir():
            continue
        if skip_degenerate and seq.name in DEGENERATE:
            continue
        if not (seq.name.startswith("i_") or seq.name.startswith("v_")):
            continue
        subset = "illum" if seq.name.startswith("i_") else "view"
        im1 = seq / "1.ppm"
        if not im1.exists():
            continue
        for k in range(2, 7):
            imk = seq / f"{k}.ppm"
            Hk = seq / f"H_1_{k}"
            if not (imk.exists() and Hk.exists()):
                continue
            H = np.loadtxt(Hk)
            yield {
                "im_A_path": str(im1),
                "im_B_path": str(imk),
                "H_gt": H,
                "subset": subset,
                "seq": seq.name,
                "pair_idx": k,
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


register("hpatches", DatasetSpec(
    task="homography",
    iter_pairs=iter_pairs,
    default_root=Path("hpatches"),
    default_resize=None,
    download=download,
    source_url=f"https://huggingface.co/datasets/{_HF_REPO}",
    license="CC BY-NC-SA 4.0",
))
