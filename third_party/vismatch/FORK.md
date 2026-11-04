# OpsiClear fork of vismatch

This is OpsiClear's fork of [gmberton/vismatch](https://github.com/gmberton/vismatch)
(BSD-3-Clause). It keeps upstream's layout so new upstream matchers and fixes merge in cleanly,
and carries a small set of OpsiClear changes on top.

**Policy: never open pull requests, issues or discussions on the upstream repository, and never push
to it.** The `upstream` remote is fetch-only (its push URL is set to `DISABLED`).

## Where it lives

- **Development checkout:** `featurexx/third_party/vismatch`, with remotes `upstream`
  (gmberton/vismatch, fetch-only).
- **Published copy:** vendored into [visbench](https://github.com/lyehe/visbench) at
  `third_party/vismatch` as a squashed `git subtree`.

## Syncing with upstream

```bash
python scripts/sync_upstream.py            # fetch + merge upstream/main
python scripts/sync_upstream.py --ref v1.5.0
```

The script merges upstream, sets `fallback_version` in `pyproject.toml` to the latest merged
upstream tag (used when the package is built from the vendored copy, which has no `.git` of its
own), and lists matchers that upstream added or removed. After a sync, initialise any new model
submodules you need and run the tests (`pytest tests`).

Then pull the fork into visbench, from the visbench root:

```bash
uv run python scripts/sync_vismatch.py --fork <path-to-this-checkout> --branch main
```

## Dependency drift in the shared development environment

CPU import checks on 2026-11-04 passed for `vismatch.im_models.edm` and
`vismatch.im_models.ripe` with setuptools **80.10.2** and lightning **2.3.3**
(Python 3.10; pytorch-lightning 2.6.1). No model construction, weights or inference
were tested. EDM imports `lightning.pytorch.utilities.rank_zero_only` through
`src.utils.misc`; RIPE's inference import does not need Lightning (its training
code uses `lightning.fabric.Fabric`). Neither inference import needs an API
introduced in Lightning 2.4 in this checkout. The installed versions are below
upstream's declared requirements, but do not currently block these imports.

`requirements.txt` declares **runtime** dependencies `lightning>=2.4` (for EDM and
RIPE) and `setuptools>=83`; the latter is not just a build requirement. The
separate build requirement in `pyproject.toml` is `setuptools>=61` plus
`setuptools-scm>=8`. No matcher-specific need for setuptools 83 was found in
the EDM/RIPE inference imports. Leave upstream's dependency declarations intact.

Do not upgrade setuptools alone: installed Lightning 2.3.3 imports
`pkg_resources` in `lightning/fabric/__init__.py`, and
[setuptools removed it in version 82](https://setuptools.pypa.io/en/stable/deprecated/pkg_resources.html).
Without it, EDM's import fails with `ModuleNotFoundError: No module named
'pkg_resources'`. To align with upstream, the exact command from the fork root
is below; it is a recommendation for a separately scheduled environment change,
and was **not run** during this audit:

```powershell
uv pip install --python .venv/Scripts/python.exe --upgrade "setuptools>=83" "lightning>=2.4"
```

This can change Lightning and transitive packages shared by other pipelines;
the resolver may also change the Torch stack. Vendored GIM and keypt2subpx
gluefactory evaluation helpers also import `pkg_resources`, so upgrading
Lightning alone does not certify every optional matcher path. Validate the resolved versions
and rerun CPU imports/tests before using that environment for benchmarks. The
current smoke tests do not certify inference or training with newer packages.

## Initialising optional matcher submodules

CLIDD, FlexGlue, dad, REDI-Match, RaCo, SANDesc and UPAL are empty in the development
checkout at this audit. `add_to_path` raises an `ImportError` naming a missing or
empty bundled repository and its selective init command, before attempting its
imports. Initialise only the models you need, from the fork root, for example:

```bash
git submodule update --init vismatch/third_party/CLIDD
```

Replace `CLIDD` with `FlexGlue`, `dad`, `REDI-Match`, `RaCo`, `SANDesc` or `UPAL`
(case matters). In visbench use the superproject root and the vendored prefix:

```bash
git submodule update --init third_party/vismatch/vismatch/third_party/CLIDD
```

Some wrappers also need other submodules: FlexGlue imports RaCo; dad-roma uses
RoMa; raco-lightglue uses LightGlue; SANDesc imports LightGlue and DeDoDe. Init
those selectively if required. Do not use `--recursive` or initialise all model
submodules. Initialising source code does not install optional dependencies or
weights, and CLIDD's kernels require CUDA when constructing the model.

## OpsiClear changes on top of upstream

Keep these small and in upstream's style, so merges stay conflict-free:

- `to_numpy` returns converted dicts (it used to raise for every dict).
- `np.trapz` is aliased to `np.trapezoid` on NumPy 2 for the vendored pose-AUC helpers.
- Docs "view source" links resolve relative to the installed package.
- `fallback_version` for builds without a git checkout.
- `add_to_path` raises a clear `ImportError` (with the init command) for an uninitialised model submodule.
- A pre-commit config: check-only hygiene hooks, plus ruff with upstream's own settings (it fixes and
  formats only the files you commit; upstream files already pass).
- `scripts/`: this sync script and local benchmark wrappers (`run_benchmark.{sh,ps1}`) that keep
  model weights in `<project>/datasets/vismatch_cache`.
