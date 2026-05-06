"""visbench CLI dispatch.

Subcommands:
  list datasets          show registered datasets (name, task, default_root)
  list methods           show registered methods + (if vismatch installed) vismatch matchers
  download <name|--all>  download / prepare a dataset
  run                    one (dataset, method) cell
  batch                  one method, many datasets
  standard               panel-based (datasets x methods) sweep
  report                 aggregate per-cell JSONs into ranking tables
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _default_data_root() -> Path:
    """Where datasets live by default. Configurable via VISBENCH_DATA_ROOT."""
    env = os.environ.get("VISBENCH_DATA_ROOT")
    if env:
        return Path(env)
    return Path.cwd() / "datasets"


def _setup_caches(data_root: Path) -> None:
    cache_root = data_root / "_cache"
    os.environ.setdefault("HF_HOME", str(cache_root / "hf"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_root / "hf" / "hub"))
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def _cmd_list(args):
    from . import datasets as _ds_pkg  # noqa: F401  (registers)
    from . import methods as _m_pkg    # noqa: F401
    from .datasets import available as ds_avail, get as ds_get
    from .methods import available as m_avail
    from .methods.vismatch_loader import list_vismatch_matchers

    if args.kind == "datasets":
        for d in ds_avail():
            spec = ds_get(d)
            print(f"  {d:<32} task={spec.task:<14} default_root={spec.default_root}")
    elif args.kind == "methods":
        print("classical:")
        for m in m_avail():
            print(f"  {m}")
        print("\nvismatch (use as 'vismatch:<name>'):")
        names = list_vismatch_matchers()
        if not names:
            print("  (vismatch not importable — install editable: uv pip install -e ./third_party/vismatch)")
        else:
            for n in names:
                print(f"  {n}")
    else:
        raise SystemExit("Usage: visbench list {datasets|methods}")


def _cmd_download(args):
    from . import datasets as _ds_pkg  # noqa: F401
    from .datasets import available as ds_avail, get as ds_get

    data_root = Path(args.data_root) if args.data_root else _default_data_root()
    _setup_caches(data_root)

    if args.all_auto:
        names = ds_avail()
    elif args.name:
        names = [args.name]
    else:
        raise SystemExit("Usage: visbench download <name> | visbench download --all-auto")

    for name in names:
        try:
            spec = ds_get(name)
        except KeyError as e:
            print(f"[skip] {name}: {e}")
            continue
        dl = spec.download
        if dl is None:
            print(f"[skip] {name}: no download() (manual data placement required)")
            continue
        target = data_root / spec.default_root
        try:
            dl(target)
            print(f"[ok]   {name} -> {target}")
        except NotImplementedError as e:
            if args.all_auto:
                print(f"[skip] {name}: {e}")
            else:
                print(f"[manual] {name}:\n{e}")
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {name}: {type(e).__name__}: {e}")


def _cmd_run(args):
    from . import datasets as _ds_pkg  # noqa: F401
    from . import methods as _m_pkg    # noqa: F401
    from .orchestrators.run import run_one

    data_root_default = Path(args.data_root_default) if args.data_root_default else _default_data_root()
    _setup_caches(data_root_default)

    device = args.device or _default_device()
    record = run_one(
        dataset=args.dataset, method=args.method,
        data_root_override=Path(args.data_root) if args.data_root else None,
        data_root_default=data_root_default,
        max_pairs=args.max_pairs, resize=args.resize,
        ransac_runs=args.ransac_runs, device=device,
        out=Path(args.out) if args.out else None,
        progress=not args.no_progress,
    )
    print(json.dumps(record, indent=2))


def _cmd_batch(args):
    from . import datasets as _ds_pkg  # noqa: F401
    from . import methods as _m_pkg    # noqa: F401
    from .orchestrators.batch import run_batch

    data_root_default = Path(args.data_root_default) if args.data_root_default else _default_data_root()
    _setup_caches(data_root_default)

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    device = args.device or _default_device()

    mp = {}
    if args.max_pairs is not None:
        for ds in datasets:
            mp[ds] = args.max_pairs

    run_batch(
        method=args.method, datasets=datasets,
        out_dir=Path(args.out_dir),
        data_root_default=data_root_default,
        max_pairs=mp, ransac_runs=args.ransac_runs, device=device,
        skip_existing=not args.no_skip,
    )


def _cmd_standard(args):
    from . import datasets as _ds_pkg  # noqa: F401
    from . import methods as _m_pkg    # noqa: F401
    from .orchestrators.standard import run_standard, run_standard_parallel

    data_root_default = Path(args.data_root_default) if args.data_root_default else _default_data_root()
    _setup_caches(data_root_default)

    panel_path = _resolve_panel(args.panel)
    methods_path = Path(args.methods)

    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
        run_standard_parallel(
            panel_path=panel_path, methods_path=methods_path,
            out_dir=Path(args.out_dir), data_root_default=data_root_default,
            devices=devices,
            max_pairs_pose=args.max_pairs_pose,
            max_pairs_homography=args.max_pairs_homography,
            ransac_runs=args.ransac_runs,
            per_method_timeout=args.per_method_timeout,
            skip_existing=not args.no_skip,
        )
        return

    device = args.device or _default_device()
    run_standard(
        panel_path=panel_path, methods_path=methods_path,
        out_dir=Path(args.out_dir), data_root_default=data_root_default,
        max_pairs_pose=args.max_pairs_pose,
        max_pairs_homography=args.max_pairs_homography,
        ransac_runs=args.ransac_runs, device=device,
        skip_existing=not args.no_skip,
    )


def _cmd_report(args):
    from .orchestrators.reporting import report
    report(Path(args.results_dir), top_k=args.top_k,
           show_reference=not args.no_reference)


def _resolve_panel(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    pkg_panel = Path(__file__).resolve().parent.parent / "panels" / f"{name_or_path}.txt"
    if pkg_panel.exists():
        return pkg_panel
    raise SystemExit(f"Panel '{name_or_path}' not found (tried {p} and {pkg_panel}).")


def _default_device() -> str:
    try:
        from vismatch import get_default_device
        return get_default_device()
    except Exception:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="visbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    # list
    pl = sub.add_parser("list", help="List datasets or methods")
    pl.add_argument("kind", choices=["datasets", "methods"])
    pl.set_defaults(func=_cmd_list)

    # download
    pd = sub.add_parser("download", help="Download / prepare a dataset")
    pd.add_argument("name", nargs="?")
    pd.add_argument("--all-auto", action="store_true",
                    help="Try every dataset's download() (skips on NotImplementedError).")
    pd.add_argument("--data-root", default=None)
    pd.set_defaults(func=_cmd_download)

    # run
    pr = sub.add_parser("run", help="Run one (dataset, method) cell")
    pr.add_argument("--dataset", required=True)
    pr.add_argument("--method", required=True,
                    help="Method name; e.g. 'sift', 'rootsift', 'vismatch:lightglue'.")
    pr.add_argument("--data-root", default=None)
    pr.add_argument("--data-root-default", default=None)
    pr.add_argument("--max-pairs", type=int, default=None)
    pr.add_argument("--resize", type=int, default=None)
    pr.add_argument("--ransac-runs", type=int, default=5)
    pr.add_argument("--device", default=None)
    pr.add_argument("--out", default=None)
    pr.add_argument("--no-progress", action="store_true")
    pr.set_defaults(func=_cmd_run)

    # batch
    pb = sub.add_parser("batch", help="Run one method across many datasets in one process")
    pb.add_argument("--method", required=True)
    pb.add_argument("--datasets", required=True, help="Comma-separated dataset names.")
    pb.add_argument("--out-dir", required=True)
    pb.add_argument("--data-root-default", default=None)
    pb.add_argument("--max-pairs", type=int, default=None)
    pb.add_argument("--ransac-runs", type=int, default=5)
    pb.add_argument("--device", default=None)
    pb.add_argument("--no-skip", action="store_true",
                    help="Re-run cells even if their JSON already exists.")
    pb.set_defaults(func=_cmd_batch)

    # standard
    ps = sub.add_parser("standard", help="Panel-based (datasets x methods) sweep")
    ps.add_argument("--panel", required=True,
                    help="Panel name ('core', 'extended') or path to a .txt of dataset names.")
    ps.add_argument("--methods", required=True,
                    help="Path to a .txt of method names (one per line).")
    ps.add_argument("--out-dir", required=True)
    ps.add_argument("--data-root-default", default=None)
    ps.add_argument("--max-pairs-pose", type=int, default=None)
    ps.add_argument("--max-pairs-homography", type=int, default=None)
    ps.add_argument("--ransac-runs", type=int, default=5)
    ps.add_argument("--device", default=None,
                    help="Single device for sequential mode (ignored when --devices is set).")
    ps.add_argument("--devices", default=None,
                    help="Comma-separated device list (e.g. 'cuda:0,cuda:1,cpu') for parallel "
                    "subprocess execution; one method per device, round-robin.")
    ps.add_argument("--per-method-timeout", type=float, default=None,
                    help="Timeout (seconds) per method subprocess in parallel mode.")
    ps.add_argument("--no-skip", action="store_true")
    ps.set_defaults(func=_cmd_standard)

    # report
    prp = sub.add_parser("report", help="Aggregate JSONs into ranking tables")
    prp.add_argument("--results-dir", required=True)
    prp.add_argument("--top-k", type=int, default=25)
    prp.add_argument("--no-reference", action="store_true",
                     help="Suppress 'ref:' rows from reference_scores.json.")
    prp.set_defaults(func=_cmd_report)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)


if __name__ == "__main__":
    main()
