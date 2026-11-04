#!/usr/bin/env bash
# Run vismatch_test.py with model weights cached under <project>/datasets/vismatch_cache,
# where <project> is two levels above this repository (e.g. featurexx/third_party/vismatch -> featurexx).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
CACHE_ROOT="$PROJECT_ROOT/datasets/vismatch_cache"

mkdir -p "$CACHE_ROOT/hf" "$CACHE_ROOT/torch"

export HF_HOME="$CACHE_ROOT/hf"
export HF_HUB_CACHE="$CACHE_ROOT/hf/hub"
export TORCH_HOME="$CACHE_ROOT/torch"
export XDG_CACHE_HOME="$CACHE_ROOT"

PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"
[ -x "$PYTHON" ] || PYTHON="$REPO_ROOT/.venv/bin/python"

echo "HF_HOME=$HF_HOME"
echo "TORCH_HOME=$TORCH_HOME"
exec "$PYTHON" "$REPO_ROOT/vismatch_test.py" "$@"
