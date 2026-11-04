# Run vismatch_test.py with model weights cached under <project>\datasets\vismatch_cache,
# where <project> is two levels above this repository (e.g. featurexx\third_party\vismatch -> featurexx).
$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$ProjectRoot = Resolve-Path (Join-Path $RepoRoot "..\..")
$CacheRoot = Join-Path $ProjectRoot "datasets\vismatch_cache"

New-Item -ItemType Directory -Force -Path (Join-Path $CacheRoot "hf") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $CacheRoot "torch") | Out-Null

$env:HF_HOME = Join-Path $CacheRoot "hf"
$env:HF_HUB_CACHE = Join-Path $CacheRoot "hf\hub"
$env:TORCH_HOME = Join-Path $CacheRoot "torch"
$env:XDG_CACHE_HOME = $CacheRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

Write-Host "HF_HOME=$($env:HF_HOME)"
Write-Host "TORCH_HOME=$($env:TORCH_HOME)"
& $Python (Join-Path $RepoRoot "vismatch_test.py") @args
