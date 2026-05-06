"""Sanity-check the visbench install.

Runs:
  1. Import every subpackage.
  2. List registered datasets + methods.
  3. (optional) `--smoke`: download HPatches, run SIFT on 5 pairs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="Also download HPatches and run a 5-pair SIFT cell.")
    p.add_argument("--data-root", default=None)
    args = p.parse_args()

    print("== Import check ==")
    import visbench  # noqa: F401
    from visbench import datasets, methods
    from visbench.core import harness  # noqa: F401
    from visbench.orchestrators import run as run_mod  # noqa: F401
    print("  OK: visbench, datasets, methods, harness, orchestrators")

    print("\n== Registered datasets ==")
    ds = datasets.available()
    print(f"  {len(ds)} registered:")
    for name in ds[:8]:
        print(f"    {name}")
    if len(ds) > 8:
        print(f"    ... +{len(ds) - 8} more")

    print("\n== Registered methods (classical) ==")
    print("  " + ", ".join(methods.available()))

    print("\n== vismatch loader ==")
    try:
        import vismatch
        print(f"  vismatch={getattr(vismatch, '__version__', '?')}")
    except Exception as e:
        print(f"  not importable: {e}")

    if not args.smoke:
        print("\n(skip smoke; pass --smoke to also run a 5-pair SIFT cell.)")
        return

    print("\n== Smoke test (HPatches + SIFT, 5 pairs) ==")
    data_root = Path(args.data_root) if args.data_root else (Path.cwd() / "datasets")
    from visbench.datasets import get as ds_get
    spec = ds_get("hpatches")
    spec.download(data_root / spec.default_root)
    from visbench.orchestrators.run import run_one
    rec = run_one(
        dataset="hpatches", method="sift",
        data_root_override=None, data_root_default=data_root,
        max_pairs=5, ransac_runs=1, device="cpu", progress=True,
    )
    print(f"\n  AUC@5 = {rec['metrics'].get('all_auc_5', 'n/a'):.3f}")
    print(f"  total_time_s = {rec['metrics']['total_time_s']:.1f}")


if __name__ == "__main__":
    main()
