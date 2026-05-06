"""Aggregate per-cell JSONs into ranking tables, with optional inline reference scores."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HP_PANEL_DEFAULT = ["hpatches", "hpatches_rot45", "hpatches_rot90", "hpatches_rot180"]
POSE_PANEL_DEFAULT = ["megadepth1500", "tum_rgbd", "blendedmvs"]

_REFERENCE_PATH = Path(__file__).resolve().parent.parent / "reference_scores.json"


def _load_results(results_dir: Path):
    out = defaultdict(dict)
    for f in results_dir.glob("*.json"):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        if not isinstance(r, dict):
            continue
        name = (r.get("method") or r.get("matcher") or "").replace("custom:", "")
        ds = r.get("dataset", "")
        out[name][ds] = r.get("metrics", {})
    return out


def _load_references() -> dict:
    """Return {dataset: {method: metrics_dict}} from packaged reference_scores.json."""
    if not _REFERENCE_PATH.exists():
        return {}
    try:
        raw = json.loads(_REFERENCE_PATH.read_text())
    except Exception:
        return {}
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def _hp(m, key="all_auc_5"):
    return m.get(key, 0) or 0


def _pose(m, key="auc_5"):
    return m.get(key, 0) or 0


def report(results_dir: Path, top_k: int = 25,
           hp_panel: list[str] | None = None,
           pose_panel: list[str] | None = None,
           show_reference: bool = True) -> None:
    hp_panel = hp_panel or HP_PANEL_DEFAULT
    pose_panel = pose_panel or POSE_PANEL_DEFAULT
    rows = _load_results(results_dir)
    refs = _load_references() if show_reference else {}

    print(f"\n# VISBENCH REPORT  ({results_dir})\n")
    print(f"Methods with at least one result: {len(rows)}")
    if refs:
        print("Reference rows (`ref:`) come from visbench/reference_scores.json — "
              "published numbers from the literature. They are not included in your "
              "leaderboard ranking.")
    print()

    print("## Per-dataset rankings\n")
    for ds in hp_panel + pose_panel:
        is_hp = ds in hp_panel
        candidates = []
        for name, dsmap in rows.items():
            if ds not in dsmap:
                continue
            metrics = dsmap[ds]
            if is_hp:
                candidates.append(("user", name,
                                   _hp(metrics, "all_auc_5"),
                                   _hp(metrics, "all_auc_10"),
                                   _hp(metrics, "all_mma_3")))
            else:
                candidates.append(("user", name,
                                   _pose(metrics, "auc_5"),
                                   _pose(metrics, "auc_10"),
                                   _pose(metrics, "auc_20")))
        candidates.sort(key=lambda r: -r[2])
        candidates = candidates[:top_k]

        # Append reference rows AFTER ranking so they don't compete in the top-k cut.
        ref_ds = refs.get(ds, {})
        if is_hp:
            for ref_name, m in ref_ds.items():
                candidates.append(("ref", ref_name,
                                   _hp(m, "all_auc_5"),
                                   _hp(m, "all_auc_10"),
                                   _hp(m, "all_mma_3")))
        else:
            for ref_name, m in ref_ds.items():
                candidates.append(("ref", ref_name,
                                   _pose(m, "auc_5"),
                                   _pose(m, "auc_10"),
                                   _pose(m, "auc_20")))

        if not candidates:
            print(f"### {ds}: (no results)\n")
            continue
        if is_hp:
            print(f"### {ds}  (homography, AUC@5/AUC@10/MMA@3)\n")
            print(f"  {'method':<45} AUC@5   AUC@10  MMA@3")
        else:
            print(f"### {ds}  (pose, AUC@5/10/20 deg)\n")
            print(f"  {'method':<45} AUC@5   AUC@10  AUC@20")
        for kind, name, a, b, c in candidates:
            label = (f"ref: {name}" if kind == "ref" else name)[:43]
            print(f"  {label:<45} {a:.3f}   {b:.3f}   {c:.3f}")
        print()

    print("\n## Combined ranking (mean across panel)\n")
    combined = []
    for name, dsmap in rows.items():
        scores = []
        for ds in hp_panel:
            if ds in dsmap:
                scores.append(_hp(dsmap[ds], "all_auc_5"))
        for ds in pose_panel:
            if ds in dsmap:
                scores.append(_pose(dsmap[ds], "auc_5"))
        if len(scores) >= max(1, len(hp_panel) // 2):
            combined.append((name, sum(scores) / len(scores), len(scores), dsmap))
    combined.sort(key=lambda x: -x[1])

    cell_names = hp_panel + pose_panel
    header = "  " + f"{'method':<48} mean   n_ds  " + "  ".join(c[:6] for c in cell_names)
    print(header)
    for name, mean, n, dsmap in combined[:top_k]:
        cells = []
        for ds in hp_panel:
            cells.append(f"{_hp(dsmap[ds], 'all_auc_5'):.3f}" if ds in dsmap else "  -  ")
        for ds in pose_panel:
            cells.append(f"{_pose(dsmap[ds], 'auc_5'):.3f}" if ds in dsmap else "  -  ")
        print(f"  {name[:46]:<48} {mean:.3f}  {n}     {' '.join(cells)}")
    print()
