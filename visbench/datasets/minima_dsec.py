"""DSEC RGB-Event synthetic-homography iterator (MINIMA protocol).

Source:  https://dsec.ifi.uzh.ch/ + MINIMA https://github.com/LSXI7/MINIMA
Paper:   Gehrig et al., 'DSEC: A Stereo Event Camera Dataset for Driving Scenarios', RAL 2021.
License: CC BY-NC-SA 4.0.

Note: DSEC's preprocessed event-image bundle is hosted on Google Drive;
gdown often hits the quota interstitial. Provide your own Google Drive file_id
via VISBENCH_DSEC_FILE_ID env var, or fetch manually and place under
`<data_root>/DSEC/` with subdirs `<seq>/images/*.png` and `<seq>/events/*.png`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from ._download import download_gdown_bypass, extract_archive
from ._minima_homography import (
    load_resized_to_disk,
    sample_homography,
    warp_and_cache,
)
from ._registry import DatasetSpec, register

TARGET_SHAPE = (480, 640)


def download(data_root: Path | str) -> Path:
    data_root = Path(data_root)
    inner = data_root / "DSEC"
    if inner.exists() and any(inner.glob("*/images/*.png")):
        return data_root
    file_id = os.environ.get("VISBENCH_DSEC_FILE_ID")
    if not file_id:
        raise NotImplementedError(
            "DSEC: set VISBENCH_DSEC_FILE_ID to the Google Drive file_id of the\n"
            "MINIMA-prepared DSEC archive (drive.google.com/file/d/<ID>/view), then\n"
            "re-run. Or place the data manually at "
            f"{data_root}/DSEC/<seq>/{{images,events}}/*.png ."
        )
    archive = data_root / "dsec.zip"
    download_gdown_bypass(file_id, archive)
    extract_archive(archive, data_root)
    try:
        archive.unlink()
    except OSError:
        pass
    return data_root


def _enumerate_pairs_from_disk(root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for seq in sorted(root.iterdir()):
        if not seq.is_dir():
            continue
        img_dir = seq / "images"
        evt_dir = seq / "events"
        if not (img_dir.exists() and evt_dir.exists()):
            continue
        for img in sorted(img_dir.glob("*.png")):
            evt = evt_dir / img.name
            if evt.exists():
                pairs.append((img, evt))
    return pairs


def _enumerate_pairs_from_list(root: Path, list_txt: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    with open(list_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vis = root / line
            evt = root / line.replace("images", "events")
            if vis.exists() and evt.exists():
                pairs.append((vis, evt))
    return pairs


def iter_pairs(data_root: Path | str, max_pairs: int | None = None) -> Iterator[dict]:
    data_root = Path(data_root)
    inner = data_root / "DSEC"
    if inner.exists():
        data_root = inner

    list_txt = data_root / "event_list.txt"
    pairs = (_enumerate_pairs_from_list(data_root, list_txt)
             if list_txt.exists() and list_txt.stat().st_size > 0
             else _enumerate_pairs_from_disk(data_root))
    if not pairs:
        raise FileNotFoundError(f"DSEC: no (image, event) pairs under {data_root}")

    cache_root = data_root / "_warp_cache"
    count = 0

    for vis_path, evt_path in pairs:
        rel_vis = vis_path.relative_to(data_root)
        rel_evt = evt_path.relative_to(data_root)

        H_vis_to_evt = sample_homography(TARGET_SHAPE, str(vis_path))
        H_evt_to_vis = sample_homography(TARGET_SHAPE, str(evt_path))

        vis_resized = load_resized_to_disk(vis_path, cache_root / "vis" / rel_vis, TARGET_SHAPE)
        evt_resized = load_resized_to_disk(evt_path, cache_root / "evt" / rel_evt, TARGET_SHAPE)
        evt_warped = warp_and_cache(
            evt_path, H=H_vis_to_evt,
            cache_path=cache_root / "evt_warped" / rel_evt, target_shape=TARGET_SHAPE,
        )
        vis_warped = warp_and_cache(
            vis_path, H=H_evt_to_vis,
            cache_path=cache_root / "vis_warped" / rel_vis, target_shape=TARGET_SHAPE,
        )

        for direction, im_A, im_B, H_gt in [
            ("vis_to_evt", vis_resized, evt_warped, H_vis_to_evt),
            ("evt_to_vis", evt_resized, vis_warped, H_evt_to_vis),
        ]:
            yield {
                "im_A_path": str(im_A),
                "im_B_path": str(im_B),
                "H_gt": H_gt,
                "subset": f"DSEC-event-{direction}",
                "seq": rel_vis.parts[0],
            }
            count += 1
            if max_pairs is not None and count >= max_pairs:
                return


register("minima_dsec", DatasetSpec(
    task="homography",
    iter_pairs=iter_pairs,
    default_root=Path("DSEC"),
    default_resize=None,
    download=download,
    source_url="https://dsec.ifi.uzh.ch/",
    license="CC BY-NC-SA 4.0",
))
