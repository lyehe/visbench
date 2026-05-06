# visbench

Benchmark harness for visual feature matching. Ships:

- **Core harness** — pose / homography / classical metrics with shared image caching, RANSAC dispatch, and AUC reporting.
- **~40 datasets** — registered behind a single interface with `download(data_root)` and `iter_pairs(data_root)`.
- **Reference methods** — OpenCV classical (SIFT, RootSIFT, ORB, AKAZE, BRISK) plus a passthrough to any matcher in [vismatch](https://github.com/gmberton/vismatch).
- **Standard CLI** — `visbench download`, `visbench run`, `visbench standard`, `visbench report`.

## Install

```bash
git clone --recurse-submodules <visbench-url>
cd visbench
uv venv
uv pip install -e .
uv pip install -e ./third_party/vismatch
```

`uv` resolves torch from the CUDA 13.0 index pinned in `pyproject.toml` (matches vismatch's stack). For CPU-only or different CUDA, edit `[[tool.uv.index]]`.

If you forgot `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

## Quickstart

```bash
visbench list datasets
visbench list methods

visbench download hpatches
visbench run --dataset hpatches --method sift --max-pairs 50 --out results/

# Run any vismatch matcher by name:
visbench run --dataset hpatches --method vismatch:lightglue --max-pairs 50

# Standard panel (multiple datasets x methods):
visbench standard --panel core --methods panels/classical_methods.txt --out-dir results/
visbench report --results-dir results/ --top 10
```

## Datasets and benchmarks

- [DATASETS.md](DATASETS.md) — full catalog: source URLs, papers, licenses, and download mechanism per dataset. Form-gated datasets (ScanNet/ScanNet++, Map-free, CMU-Seasons, Oxford-RobotCar, Tanks&Temples, nuScenes/RUBIK, MegaDepth-X) are listed there but not shipped.
- [BENCHMARKS.md](BENCHMARKS.md) — canonical protocol per dataset (resize / RANSAC / metric) plus published reference scores from the literature (RoMa, LoFTR, LightGlue, SIFT-NN, RootSIFT, …) so visbench JSONs are paper-comparable.

## License

BSD 3-Clause — see [LICENSE](LICENSE).

The vendored `vismatch` submodule is BSD 3-Clause (Copyright 2024 Alex Stoken, Gabriele Berton). Each dataset has its own license; consult the source linked in `DATASETS.md` and in each dataset module's header docstring before redistributing data.
