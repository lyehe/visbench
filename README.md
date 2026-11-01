# visbench

Benchmark harness for visual feature matching:

- **Benchmarks recent papers report** — MegaDepth-1500, HardMatch, WxBS, ETH3D,
  HPatches, SatAst, the MINIMA cross-modal suite, rotated splits and a compact
  stress set (35 datasets, all public; see [DATASETS.md](DATASETS.md)).
- **Reference protocols** — each task reproduces its paper's evaluator (pose
  AUC, WxBS/HardMatch mAA, HPatches corner AUC, SatAst reprojection AUC), with
  published scores alongside ([BENCHMARKS.md](BENCHMARKS.md)).
- **Methods** — OpenCV classical (SIFT, RootSIFT, ORB, AKAZE, BRISK), any
  [vismatch](https://github.com/gmberton/vismatch) matcher (`vismatch:<name>`,
  including LoMa-B128/B/L/G), and optional RoMaXX TensorRT matchers
  (`romaxx:<kind>[:<arg>]`).
- **CLI** — `list`, `download`, `check`, `run`, `batch`, `standard`, `report`.

## Install

```bash
git clone https://github.com/lyehe/visbench
cd visbench

# Pull only the vismatch submodule (one level deep). vismatch itself has 25+
# nested matcher submodules that are NOT needed at install time; it loads them
# lazily on first use.
git submodule update --init third_party/vismatch

uv venv
uv pip install -e .
uv pip install -e ./third_party/vismatch
```

`uv` resolves torch from the CUDA 13.0 index pinned in `pyproject.toml`. For
CPU-only or another CUDA version, edit `[[tool.uv.index]]`.

> **Do not use `--recurse-submodules` or `git submodule update --init --recursive`.**
> That clones every matcher repo nested inside vismatch (tens of GB).

Optional extras: `pip install -e .[aachen]` (pycolmap for Aachen), and RoMaXX
installed into the same environment for `romaxx:` methods.

## Quickstart

```bash
visbench list datasets
visbench list methods

visbench download hardmatch
visbench check hardmatch                 # sanity-check the ground truth on CPU
visbench run --dataset hardmatch --method vismatch:loma-g --out results/hardmatch__loma-g.json

# Core panel (MegaDepth-1500, HardMatch, WxBS, ETH3D, HPatches) x methods:
visbench standard --panel core --methods panels/sota_methods.txt --out-dir results/
visbench report --results-dir results/            # rankings, leaderboard, matrix
```

Panels: `core`, `extended` (rotated and harder splits, SatAst, Aachen),
`crossmodal`, `stress`. Method lists: `panels/*_methods.txt`.

Data lives under `$VISBENCH_DATA_ROOT` (default `./datasets`). Set
`VISBENCH_KEEP_CACHES=1` to stop visbench from pointing the HF/torch caches
under the data root.

## Layout

```text
visbench/
  core/            harness (eval_* per task), metrics, pair sampling, GT check
  datasets/        pose/ correspondence/ homography/ crossmodal/ stress/
  methods/         classical, vismatch passthrough, romaxx adapter
  orchestrators/   run, batch, standard, report, check
  reference_scores.json
panels/            dataset and method lists
```

## License

BSD 3-Clause — see [LICENSE](LICENSE).

The vendored `vismatch` submodule is BSD 3-Clause (Copyright 2024 Alex Stoken,
Gabriele Berton). Each dataset has its own license; consult the source linked
in `DATASETS.md` and in each dataset module's docstring before redistributing
data.
