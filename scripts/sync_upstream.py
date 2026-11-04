"""Merge new upstream vismatch (gmberton/vismatch) into this OpsiClear fork.

The fork keeps upstream's layout, so syncing is a normal merge. This script:

1. fetches the ``upstream`` remote (fetch-only: its push URL is disabled, and we never open PRs or
   issues upstream),
2. merges ``upstream/<branch>`` into the current branch (stops and lists conflicts if any),
3. sets ``fallback_version`` in pyproject.toml to the latest merged upstream tag, and
4. prints the matchers that upstream added or removed.

Usage, from the repo root on the fork's main branch::

    python scripts/sync_upstream.py            # merge upstream/main
    python scripts/sync_upstream.py --ref v1.5.0

Afterwards: initialise any new model submodules you need, run the tests, and pull the fork into
visbench with ``scripts/sync_vismatch.py`` there.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True, text=True)


def available_models(rev: str) -> list[str]:
    src = git("show", f"{rev}:vismatch/__init__.py").stdout
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "available_models":
            return [e.value for e in node.value.elts]
    return []


def set_fallback_version(tag: str) -> bool:
    path = ROOT / "pyproject.toml"
    text = path.read_text(encoding="utf-8")
    new = re.sub(r'(?m)^fallback_version = ".*"$', f'fallback_version = "{tag.lstrip("v")}+opsiclear"', text)
    if new == text:
        return False
    path.write_text(new, encoding="utf-8", newline="\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default="upstream/main", help="upstream ref to merge (default: upstream/main)")
    args = ap.parse_args()

    if git("status", "--porcelain", "--untracked-files=no", "--ignore-submodules=dirty").stdout.strip():
        sys.exit("working tree has uncommitted changes (outside submodules); commit or stash them first")

    git("fetch", "--tags", "upstream")
    start = git("rev-parse", "HEAD").stdout.strip()
    before = available_models(start)
    merge = git("merge", "--no-edit", "-m", f"Merge {args.ref} into the OpsiClear fork", args.ref, check=False)
    if merge.returncode != 0:
        conflicts = git("diff", "--name-only", "--diff-filter=U").stdout.split()
        print(merge.stdout + merge.stderr)
        sys.exit(
            f"merge stopped with conflicts in: {conflicts}\nresolve them, commit, then re-run for the version bump"
        )

    tag = git("describe", "--tags", "--abbrev=0", args.ref).stdout.strip()
    if set_fallback_version(tag):
        git("add", "pyproject.toml")
        git("commit", "-m", f"chore: set fallback_version to upstream {tag}")

    after = available_models("HEAD")
    added = [m for m in after if m not in before]
    removed = [m for m in before if m not in after]
    print(f"merged {args.ref} (latest upstream tag {tag})")
    print(f"new matchers: {added or 'none'}")
    if removed:
        print(f"removed matchers: {removed}")
    if git("diff", "--quiet", start, "HEAD", "--", ".gitmodules", check=False).returncode:
        print("upstream changed .gitmodules: run `git submodule status` and init the new model submodules you need")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
