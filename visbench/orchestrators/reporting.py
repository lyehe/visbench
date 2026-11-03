"""Aggregate per-cell JSONs into ranking tables, with optional inline reference scores.

Each dataset is reported with the metrics of its own task (looked up in the
dataset registry), so a homography dataset never shows pose keys as zeros. A
metric a record does not contain is shown as ``n/a`` and excluded from means.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_REFERENCE_PATH = Path(__file__).resolve().parent.parent / "reference_scores.json"
_PANEL_DIR = Path(__file__).resolve().parent.parent.parent / "panels"

# task -> [(metric key, column label)]; the first entry is the ranking metric.
TASK_METRICS: dict[str, list[tuple[str, str]]] = {
    "pose": [("auc_5", "AUC@5"), ("auc_10", "AUC@10"), ("auc_20", "AUC@20")],
    "homography": [("all_auc_3", "AUC@3px"), ("all_auc_5", "AUC@5px"), ("all_auc_10", "AUC@10px")],
    "homography_points": [("auc_10px", "AUC@10px"), ("auc_5px", "AUC@5px"), ("auc_20px", "AUC@20px")],
    "correspondence": [("all_mAA_10", "mAA@10px"), ("all_mAA_5", "mAA@5px"), ("all_pck_10", "PCK@10px")],
}


def _canonical_method(name: str) -> str:
    """One key per method: `romaxx:kind:/c/dir\\x` and `romaxx:kind:C:/dir/x` are the
    same bundle written from Git Bash vs a native path, so results must merge."""
    if not name.startswith("romaxx:"):
        return name
    kind, sep, arg = name.split(":", 1)[1].partition(":")
    if not sep:
        return name
    arg = arg.replace("\\", "/").rstrip("/")
    if len(arg) > 2 and arg[0] == "/" and arg[2] == "/" and arg[1].isalpha():
        arg = f"{arg[1]}:{arg[2:]}"
    if len(arg) > 1 and arg[1] == ":":
        arg = arg[0].upper() + arg[1:]
    return f"romaxx:{kind}:{arg}"


def _load_results(results_dir: Path):
    """Read every per-cell JSON in results_dir; skip files that aren't a
    valid record (corrupted JSON, missing method/dataset, wrong shape).

    Records without `method` AND without `dataset` are silently skipped — they
    likely came from a different tool. Records with only one of those will
    bucket into "" (visible as an empty row in the report) so the user can
    notice the malformed file rather than getting silent data loss.
    """
    out = defaultdict(dict)
    for f in sorted(results_dir.glob("*.json")):
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(r, dict):
            continue
        name = _canonical_method((r.get("method") or r.get("matcher") or "").replace("custom:", ""))
        ds = r.get("dataset", "")
        if not name and not ds:
            continue
        out[name][ds] = r.get("metrics", {}) or {}
    return out


def _load_references() -> tuple[dict, dict, dict]:
    """Return (refs, aliases, no_published) from packaged reference_scores.json."""
    if not _REFERENCE_PATH.exists():
        return {}, {}, {}
    try:
        raw = json.loads(_REFERENCE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, {}, {}
    aliases = raw.get("_aliases", {}) or {}
    no_pub = raw.get("_no_published_reference", {}) or {}
    refs = {k: v for k, v in raw.items() if not k.startswith("_")}
    return refs, aliases, no_pub


def _resolve_ref(ds: str, refs: dict, aliases: dict, no_published: dict
                 ) -> tuple[dict | None, str | None]:
    """Returns (method_metrics_map, status_note). Follows alias chains."""
    seen = set()
    cur = ds
    while cur in aliases and cur not in seen:
        seen.add(cur)
        cur = aliases[cur]
    if cur in refs:
        note = f"inherits from `{cur}`" if cur != ds else None
        return refs[cur], note
    if ds in no_published:
        return None, no_published[ds]
    return None, None


def dataset_task(ds: str, sample_metrics: dict | None = None) -> str:
    """Task from the registry; falls back to metric keys for unregistered datasets."""
    try:
        from ..datasets import get as get_dataset_spec
        return get_dataset_spec(ds).task
    except Exception:  # noqa: BLE001
        keys = set(sample_metrics or {})
        for task, cols in TASK_METRICS.items():
            if cols[0][0] in keys:
                return task
        return "pose"


def _value(metrics: dict, key: str):
    v = metrics.get(key)
    return float(v) if isinstance(v, (int, float)) else None


def short_name(method: str) -> str:
    """Compact method label for tables: `vismatch:loma-g` -> `loma-g`,
    `romaxx:flexglue-trt:<dir>` -> `flexglue-trt:<dir name>`."""
    if method.startswith("vismatch:"):
        return method.split(":", 1)[1]
    if method.startswith("romaxx:"):
        kind, _, arg = method.split(":", 1)[1].partition(":")
        if "/" in arg or "\\" in arg:
            arg = Path(arg.replace("\\", "/").rstrip("/")).name
        return f"{kind}:{arg}" if arg else kind
    return method


def _fmt(v) -> str:
    return f"{v:.3f}" if v is not None else " n/a "


def default_panel() -> list[str]:
    path = _PANEL_DIR / "core.txt"
    if not path.exists():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")]


def report(results_dir: Path, top_k: int = 25,
           panel: list[str] | None = None,
           show_reference: bool = True,
           json_out: Path | None = None,
           matrix_metric: str | None = None) -> dict:
    """Print per-dataset rankings, a combined leaderboard and a methods x datasets matrix.

    `panel` selects and orders the datasets (default: panels/core.txt, else every
    dataset present). The combined ranking averages each dataset's ranking
    metric over the panel datasets a method has results for (at least half).
    `matrix_metric` overrides the matrix cell metric (default: each dataset's
    ranking metric). Returns the structured payload; `json_out` also writes it.
    """
    rows = _load_results(Path(results_dir))
    present = sorted({ds for dsmap in rows.values() for ds in dsmap})
    panel = list(panel) if panel else [ds for ds in default_panel() if ds in present] or present
    refs, aliases, no_published = (_load_references() if show_reference else ({}, {}, {}))
    payload: dict = {"results_dir": str(results_dir), "panel": panel,
                     "per_dataset": {}, "combined": [], "matrix": {}}

    print(f"\n# VISBENCH REPORT  ({results_dir})\n")
    print(f"Methods with at least one result: {len(rows)}")
    if refs:
        print("`ref:` rows are published numbers from visbench/reference_scores.json; "
              "they are not ranked against your results.")
    print("\n## Per-dataset rankings\n")

    tasks: dict[str, str] = {}
    for ds in panel:
        sample = next((dsmap[ds] for dsmap in rows.values() if ds in dsmap), None)
        task = dataset_task(ds, sample)
        tasks[ds] = task
        cols = TASK_METRICS[task]
        ranked = []
        for name, dsmap in rows.items():
            if ds in dsmap:
                ranked.append(("user", name, [_value(dsmap[ds], k) for k, _ in cols]))
        ranked.sort(key=lambda r: -(r[2][0] if r[2][0] is not None else float("-inf")))
        ranked = ranked[:top_k]
        ref_ds, ref_note = _resolve_ref(ds, refs, aliases, no_published)
        for ref_name, m in (ref_ds or {}).items():
            ranked.append(("ref", ref_name, [_value(m, k) for k, _ in cols]))
        if not ranked:
            print(f"### {ds}: (no results)\n")
            continue
        labels = [label for _, label in cols]
        print(f"### {ds}  ({task}: {' / '.join(labels)})\n")
        print(f"  {'method':<45} " + "  ".join(f"{label:>8}" for label in labels))
        for kind, name, vals in ranked:
            label = (f"ref: {name}" if kind == "ref" else short_name(name))[:43]
            print(f"  {label:<45} " + "  ".join(f"{_fmt(v):>8}" for v in vals))
        if show_reference and ref_note:
            prefix = "(no published reference) -- " if not ref_ds else ""
            print(f"  ref: {prefix}{ref_note[:80]}")
        print()
        payload["per_dataset"][ds] = {
            "task": task, "metric_labels": labels,
            "rows": [{"kind": k, "method": n, "values": v} for k, n, v in ranked],
        }

    print("\n## Combined ranking (mean ranking metric across panel)\n")
    combined = []
    for name, dsmap in rows.items():
        cells = []
        for ds in panel:
            key = TASK_METRICS[tasks[ds]][0][0]
            cells.append(_value(dsmap[ds], key) if ds in dsmap else None)
        scores = [v for v in cells if v is not None]
        if scores and len(scores) >= max(1, len(panel) // 2):
            combined.append((name, sum(scores) / len(scores), len(scores), cells))
    combined.sort(key=lambda x: -x[1])
    print("  " + f"{'method':<46} mean   n_ds  " + "  ".join(ds[:8] for ds in panel))
    for name, mean, n, cells in combined[:top_k]:
        print(f"  {short_name(name)[:44]:<46} {mean:.3f}  {n:<4}  " + "  ".join(f"{_fmt(v):>8}" for v in cells))
        payload["combined"].append({"method": name, "mean": mean, "n_datasets": n,
                                    "cells": [{"dataset": ds, "value": v} for ds, v in zip(panel, cells)]})

    # Datasets as rows (full names), methods as columns: panels are long and
    # method lists short, so this keeps every dataset name readable.
    print(f"\n## Matrix ({matrix_metric or 'ranking metric per dataset'}; rows = datasets)\n")
    methods = sorted(rows)
    width = max([len(ds) for ds in panel] + [8])
    labels = [short_name(m) for m in methods]
    col = max([len(lb) for lb in labels] + [8])
    print("  " + f"{'dataset':<{width}} " + "  ".join(f"{lb:>{col}}" for lb in labels))
    for name in methods:
        payload["matrix"][name] = {}
    for ds in panel:
        key = matrix_metric or TASK_METRICS[tasks[ds]][0][0]
        vals = []
        for name in methods:
            v = _value(rows[name][ds], key) if ds in rows[name] else None
            payload["matrix"][name][ds] = v
            vals.append(v)
        print("  " + f"{ds:<{width}} " + "  ".join(f"{_fmt(v):>{col}}" for v in vals))
    print()

    if json_out is not None:
        json_out = Path(json_out)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {json_out}")
    return payload
