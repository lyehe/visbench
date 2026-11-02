# visbench

Benchmark harness for visual feature matching:

- **Benchmarks recent papers report** — MegaDepth-1500, HardMatch, WxBS, ETH3D,
  HPatches, SatAst, the MINIMA cross-modal suite, rotated splits and a compact
  stress set (31 datasets, all public; see [DATASETS.md](DATASETS.md)).
- **Reference protocols** — each task reproduces its paper's evaluator (pose
  AUC, WxBS/HardMatch mAA, HPatches corner AUC, SatAst reprojection AUC), with
  published scores alongside ([BENCHMARKS.md](BENCHMARKS.md)).
- **Methods** — OpenCV classical (SIFT, RootSIFT, ORB, AKAZE, BRISK), any
  [vismatch](https://github.com/gmberton/vismatch) matcher (`vismatch:<name>`,
  including LoMa-B128/B/L/G), and optional RoMaXX matchers
  (`romaxx:<kind>[:<arg>]`: FlexGlue, FlexGlueTRT, RoMaTRT, MinimaRoMaTRT, LoMaTRT).
- **CLI** — `list`, `download`, `check`, `run`, `batch`, `standard`, `report`.

## Install

```bash
git clone https://github.com/lyehe/visbench
cd visbench

# Pull only the vismatch submodule (one level deep). vismatch itself has 25+
# nested matcher submodules; check out only the ones you use (below).
git submodule update --init third_party/vismatch

uv venv
uv pip install -e .
uv pip install -e ./third_party/vismatch
```

`uv` resolves torch from the CUDA 13.0 index pinned in `pyproject.toml`. For
CPU-only or another CUDA version, edit `[[tool.uv.index]]`.

> **Do not use `--recurse-submodules` or `git submodule update --init --recursive`.**
> That clones every matcher repo nested inside vismatch (tens of GB).

Matcher code lives in vismatch's own submodules. Check out the ones you need,
for example LoMa:

```bash
git -C third_party/vismatch submodule update --init vismatch/third_party/LoMa
```

Optional:

- **RoMaXX** (`romaxx:` methods) — RoMaXX is not on PyPI and pins its own torch
  (FlexGlue bundles require Torch 2.11 + TensorRT 10.16). Run `romaxx:` methods
  from RoMaXX's environment with visbench on the path; vismatch is not needed
  there:

  ```bash
  PYTHONPATH=/path/to/visbench /path/to/romaxx/.venv/Scripts/python -m visbench run \
      --dataset hardmatch --method romaxx:flexglue-trt:/path/to/flexglue_native_512_core
  ```

- **SatAst images** are served only over rsync. Put `rsync` on `PATH` and run
  `visbench download satast`. On Windows, Git for Windows supplies the MSYS
  runtime but not rsync itself; MSYS2's `rsync` package (plus its `popt`,
  `libidn2`, `libunistring`, `libxxhash`, `liblz4`, `libzstd` DLLs) works.

## Quickstart

```bash
visbench list datasets
visbench list methods

visbench download hardmatch
visbench check hardmatch                 # sanity-check the ground truth on CPU
visbench run --dataset hardmatch --method vismatch:loma-g --out results/hardmatch__loma-g.json

# Core panel (MegaDepth-1500, HardMatch, WxBS, HPatches, SatAst) x methods:
visbench standard --panel core --methods panels/sota_methods.txt --out-dir results/
visbench report --results-dir results/            # rankings, leaderboard, matrix
```

Panels: `core`, `extended` (adds rotated splits and ETH3D),
`crossmodal`, `stress`. Method lists: `panels/*_methods.txt`.

`--max-pairs` (or `--max-pairs-pose` / `--max-pairs-homography`) evaluates a
seeded, scene-balanced subset. `batch` and `standard` skip a cell only when an
existing result has the same schema, pair cap and seed; `--rerun` forces it.

## Configuration

| Variable | Effect |
|---|---|
| `VISBENCH_DATA_ROOT` | Base data directory (default `./datasets`). |
| `VISBENCH_KEEP_CACHES=1` | Leave `HF_HOME` / `TORCH_HOME` alone instead of pointing them under the data root. |
| `VISBENCH_ROMAXX_NUM_CORRESP` | Matches sampled from RoMaXX dense matchers (default 5000). |
| `VISBENCH_LOMATRT_SIZE` | Input size LoMaTRT images are resized to; must match its built engines (default 512). |
| `VISBENCH_STRICT_IMPORT=1` | Fail on any dataset module import error instead of warning. |

On Windows keep the data root short: SatAst file names are about 150
characters, and paths over 260 characters are invisible to most tools unless
long paths are enabled.

## Development

```bash
uv pip install -e .[dev]
python -m pytest            # tests/ only (pyproject testpaths)
ruff check visbench tests scripts
```

Real-data tests are opt-in: `VISBENCH_SMOKE=1 VISBENCH_DATA_ROOT=<root>` runs a
two-pair smoke test per dataset present under the root, and
`tests/test_real_pose_geometry.py` checks pose ground truth against SIFT
epipolar error when the data is found.

## Layout

```text
visbench/
  core/            harness (eval_* per task), metrics, pair sampling, GT check, image IO
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
