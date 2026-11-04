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

## OpsiClear changes on top of upstream

Keep these small and in upstream's style, so merges stay conflict-free:

- `to_numpy` returns converted dicts (it used to raise for every dict).
- `np.trapz` is aliased to `np.trapezoid` on NumPy 2 for the vendored pose-AUC helpers.
- Docs "view source" links resolve relative to the installed package.
- `fallback_version` for builds without a git checkout.
- A pre-commit config that uses upstream's own ruff settings and check-only hooks.
- `scripts/`: this sync script and local benchmark wrappers (`run_benchmark.{sh,ps1}`) that keep
  model weights in `<project>/datasets/vismatch_cache`.
