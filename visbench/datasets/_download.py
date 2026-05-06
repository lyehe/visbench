"""Shared dataset-download primitives.

Each dataset module's `download(data_root)` calls into these helpers so
per-dataset code stays small and consistent.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from tqdm import tqdm


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dest: str | Path, sha256: str | None = None,
                 chunk_size: int = 1 << 16, headers: dict | None = None) -> Path:
    """Stream a URL to disk with tqdm + optional sha256 verify. Resumable via .part."""
    dest = Path(dest)
    if dest.exists() and (sha256 is None or _sha256(dest) == sha256):
        return dest
    _ensure_dir(dest.parent)
    part = dest.with_suffix(dest.suffix + ".part")
    req_headers = {"User-Agent": "visbench/0.1"}
    if headers:
        req_headers.update(headers)
    req = Request(url, headers=req_headers)
    with urlopen(req) as r:
        total = int(r.headers.get("Content-Length", 0)) or None
        with open(part, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                          desc=dest.name) as bar:
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                bar.update(len(chunk))
    if sha256 is not None:
        got = _sha256(part)
        if got != sha256:
            part.unlink(missing_ok=True)
            raise ValueError(f"sha256 mismatch for {url}: expected {sha256}, got {got}")
    part.replace(dest)
    return dest


def download_hf(repo_id: str, dest: str | Path, repo_type: str = "dataset",
                allow_patterns: list[str] | None = None,
                ignore_patterns: list[str] | None = None,
                revision: str | None = None) -> Path:
    """Wrap `huggingface_hub.snapshot_download` with hf_transfer enabled by default."""
    from huggingface_hub import snapshot_download

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    dest = _ensure_dir(Path(dest))
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        revision=revision,
    )
    return dest


def download_gdown(file_id: str, dest: str | Path, fuzzy: bool = True) -> Path:
    """Primary path: standard gdown."""
    import gdown

    dest = Path(dest)
    _ensure_dir(dest.parent)
    if dest.exists():
        return dest
    url = f"https://drive.google.com/uc?id={file_id}"
    out = gdown.download(url, str(dest), fuzzy=fuzzy, quiet=False)
    if out is None:
        raise RuntimeError(f"gdown failed for file_id={file_id} (try download_gdown_bypass)")
    return Path(out)


def download_gdown_bypass(file_id: str, dest: str | Path) -> Path:
    """Fallback: curl `drive.usercontent.google.com` to dodge quota / virus-scan interstitial.

    This is the trick used in featurexx for DSEC after gdown started failing on quota.
    """
    dest = Path(dest)
    _ensure_dir(dest.parent)
    if dest.exists():
        return dest
    url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
    cmd = ["curl", "-L", "-f", "-o", str(dest), url]
    subprocess.run(cmd, check=True)
    return dest


def extract_archive(path: str | Path, dest: str | Path | None = None,
                    strip_components: int = 0) -> Path:
    """Extract `.tar` / `.tar.gz` / `.tgz` / `.zip` / `.7z`. Returns destination dir."""
    path = Path(path)
    dest = _ensure_dir(Path(dest) if dest else path.parent)
    name = path.name.lower()

    if name.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar")):
        with tarfile.open(path) as tf:
            if strip_components:
                members = []
                for m in tf.getmembers():
                    parts = m.name.split("/")
                    if len(parts) <= strip_components:
                        continue
                    m.name = "/".join(parts[strip_components:])
                    members.append(m)
                tf.extractall(dest, members=members)
            else:
                tf.extractall(dest)
    elif name.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
    elif name.endswith(".7z"):
        try:
            import py7zr
            with py7zr.SevenZipFile(path) as sz:
                sz.extractall(dest)
        except ImportError as e:
            raise ImportError("py7zr required for .7z archives; pip install py7zr") from e
    else:
        raise ValueError(f"Unsupported archive: {path}")
    return dest


def is_prepared(marker: Path) -> bool:
    """Lightweight idempotency check — used by dataset.download() to skip rework."""
    return marker.exists()


def write_marker(marker: Path, content: str = "ok") -> None:
    _ensure_dir(marker.parent)
    marker.write_text(content)


def copytree_into(src: Path, dst: Path) -> None:
    """Merge `src` directory contents into existing `dst`."""
    src = Path(src); dst = Path(dst)
    _ensure_dir(dst)
    for p in src.iterdir():
        target = dst / p.name
        if p.is_dir():
            copytree_into(p, target)
        else:
            shutil.copy2(p, target)
