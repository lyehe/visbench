"""ZEB — Zero-shot Evaluation Benchmark (12 sub-datasets, GIM).

Source:  https://github.com/xuelunshen/gim (see TEST_GIM_*.sh pipelines)
Paper:   Shen et al., 'GIM: Learning Generalizable Image Matcher From Internet Videos', ICLR 2024.
License: per-subset (academic); aggregator distributed via OneDrive/SharePoint.

Each registered as `zeb_<subset>` so users can slice by domain. The combined
`zeb_all` iterates every available subset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

from ._registry import DatasetSpec, register

ZEB_SUBSETS = (
    "GL3D", "BlendedMVS", "ETH3D_Indoor", "ETH3D_Outdoor",
    "KITTI", "RobotCar_Weather", "RobotCar_Season", "RobotCar_Night",
    "MultiFoV", "SceneNet", "ICL-NUIM", "GTA-SfM",
)

# On-disk folder names of the GIM ZEB bundle (third_party/gim/zeb, matcher_training/zeb)
# and the separator GIM uses between scene id and image name in that folder.
ZEB_DIRS = {
    "GL3D": ("GL3D", "_"), "BlendedMVS": ("BlendedMVS", "_"),
    "ETH3D_Indoor": ("ETH3DI", "-"), "ETH3D_Outdoor": ("ETH3DO", "-"),
    "KITTI": ("KITTI", "-"), "RobotCar_Weather": ("RobotcarWeather", "_"),
    "RobotCar_Season": ("RobotcarSeason", "_"), "RobotCar_Night": ("RobotcarNight", "_"),
    "MultiFoV": ("MultiFoV", "-"), "SceneNet": ("SceneNet", "-"),
    "ICL-NUIM": ("ICLNUIM", "-"), "GTA-SfM": ("GTASfM", "-"),
}


def _subset_root(data_root: Path, subset: str) -> Path:
    """<data_root>/<subset> if present, else the GIM on-disk folder name."""
    for name in (subset, ZEB_DIRS[subset][0]):
        if (data_root / name).is_dir():
            return data_root / name
    return data_root / ZEB_DIRS[subset][0]


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    if any((data_root / s).exists() or (data_root / ZEB_DIRS[s][0]).exists() for s in ZEB_SUBSETS):
        return data_root
    raise NotImplementedError(
        "ZEB: download the GIM ZEB bundle from\n"
        "  https://stuxmueducn-my.sharepoint.com/:f:/g/personal/lizijun_stu_xmu_edu_cn/"
        "EmHLjQpbpDRKmiED88rxGl4BFIkSp7vAzXifwXtvVbQA9w?e=ey8WVk\n"
        f"and unzip each sequence (GL3D/, ETH3DI/, RobotcarNight/, ...) under {data_root}.\n"
        "See https://github.com/xuelunshen/gim for the canonical layout."
    )


def _parse_line(line: str):
    toks = line.strip().split()
    if len(toks) < 36:
        return None
    imA, imB = toks[0], toks[1]
    K0 = np.array([float(x) for x in toks[2:11]]).reshape(3, 3)
    K1 = np.array([float(x) for x in toks[11:20]]).reshape(3, 3)
    T = np.array([float(x) for x in toks[20:36]]).reshape(4, 4)
    return imA, imB, K0, K1, T


def _iter_gim_flat(subset: str, sub_root: Path, max_pairs: int | None) -> Iterator[dict]:
    """GIM's native layout: one ``<scene><sep>*.txt`` per pair holding
    ``img0 img1 covis0 covis1 K0(9) K1(9) T_0to1(16)``; images are
    ``<scene><sep><img stem>.png`` in the same flat folder."""
    sep = ZEB_DIRS[subset][1]
    count = 0
    for txt in sorted(sub_root.glob("*.txt")):
        toks = txt.read_text().split()
        if len(toks) < 38:
            continue
        scene_id = txt.stem.split(sep)[0]
        paths = []
        for name in toks[:2]:
            cands = [sub_root / f"{scene_id}{sep}{Path(name).stem}.png",
                     sub_root / f"{scene_id}{sep}{name}.png"]
            paths.append(next((c for c in cands if c.exists()), cands[0]))
        T = np.array([float(x) for x in toks[22:38]]).reshape(4, 4)
        yield {
            "im_A_path": str(paths[0]),
            "im_B_path": str(paths[1]),
            "K0": np.array([float(x) for x in toks[4:13]]).reshape(3, 3),
            "K1": np.array([float(x) for x in toks[13:22]]).reshape(3, 3),
            "R_0to1": T[:3, :3], "t_0to1": T[:3, 3],
            "subset": subset,
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


def iter_pairs_for_subset(subset: str, data_root: Path | str,
                          max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    sub_root = _subset_root(data_root, subset)
    pairs_file = sub_root / "pairs" / "pairs.txt"
    if not pairs_file.exists():
        pairs_file = sub_root / "pairs.txt"
    if not pairs_file.exists():
        if sub_root.is_dir() and any(sub_root.glob("*.txt")):
            yield from _iter_gim_flat(subset, sub_root, max_pairs)
            return
        raise FileNotFoundError(
            f"ZEB subset '{subset}' not ingested - expected {pairs_file} or GIM per-pair "
            f"*.txt files in {sub_root}.")
    count = 0
    for line in pairs_file.read_text().splitlines():
        parsed = _parse_line(line)
        if parsed is None:
            continue
        imA, imB, K0, K1, T = parsed
        yield {
            "im_A_path": str(sub_root / imA),
            "im_B_path": str(sub_root / imB),
            "K0": K0, "K1": K1,
            "R_0to1": T[:3, :3], "t_0to1": T[:3, 3],
            "subset": subset,
        }
        count += 1
        if max_pairs is not None and count >= max_pairs:
            return


def _make_iter(subset_name: str):
    def it(data_root, max_pairs=None, **_):
        return iter_pairs_for_subset(subset_name, data_root, max_pairs)
    return it


_KW = dict(
    download=download,
    source_url="https://github.com/xuelunshen/gim",
    license="Per-subset academic; aggregator via SharePoint",
)


for _sub in ZEB_SUBSETS:
    register(f"zeb_{_sub.lower().replace('-', '_')}", DatasetSpec(
        "pose", _make_iter(_sub), Path("zeb"), default_resize=1200, **_KW,
    ))


def iter_pairs_all(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    per_sub = None
    if max_pairs is not None:
        per_sub = max(1, max_pairs // len(ZEB_SUBSETS))
    for sub in ZEB_SUBSETS:
        try:
            yield from iter_pairs_for_subset(sub, data_root, per_sub)
        except FileNotFoundError:
            continue


register("zeb_all", DatasetSpec(
    "pose", iter_pairs_all, Path("zeb"), default_resize=1200, **_KW,
))
