"""Bootstrap every dataset whose download() is fully automatic.

Datasets whose download() raises NotImplementedError (manual / DUA-gated) are
listed at the end with their source URL so you can grab them yourself.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=str(Path.cwd() / "datasets"))
    args = p.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    from visbench.datasets import available, get
    auto, manual, missing_dl = [], [], []

    for name in available():
        spec = get(name)
        if spec.download is None:
            missing_dl.append((name, spec.source_url))
            continue
        target = data_root / spec.default_root
        try:
            print(f"[try] {name} -> {target}")
            spec.download(target)
            auto.append(name)
            print(f"[ok]  {name}")
        except NotImplementedError as e:
            manual.append((name, str(e), spec.source_url))
            print(f"[manual] {name}: {spec.source_url}")
        except Exception as e:  # noqa: BLE001
            print(f"[fail] {name}: {type(e).__name__}: {e}")

    print(f"\n=== Done. auto={len(auto)} manual={len(manual)} no_download={len(missing_dl)} ===\n")
    if manual:
        print("Manual datasets — follow the instruction printed by `visbench download <name>`:")
        for n, _msg, url in manual:
            print(f"  {n:<24} {url}")


if __name__ == "__main__":
    main()
