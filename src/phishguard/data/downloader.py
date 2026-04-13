"""Download datasets from Kaggle into data/raw/ using kagglehub."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def download_dataset(dataset_slug: str, dest_dir: str | os.PathLike) -> Path:
    """Download a Kaggle dataset to *dest_dir* via kagglehub (no auth needed).

    Parameters
    ----------
    dataset_slug:
        Kaggle dataset identifier in ``owner/dataset-name`` format.
    dest_dir:
        Local directory where files are copied after download.

    Returns
    -------
    Path
        The destination directory containing the dataset files.
    """
    import kagglehub

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"[downloader] Downloading {dataset_slug} via kagglehub …")
    cache_path = Path(kagglehub.dataset_download(dataset_slug))
    print(f"[downloader] kagglehub cache path: {cache_path}")

    # Copy every file from the cache directory into dest_dir
    for src_file in cache_path.rglob("*"):
        if src_file.is_file():
            rel = src_file.relative_to(cache_path)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, target)

    print(f"[downloader] Done. Files in {dest}:")
    for f in sorted(dest.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(dest)}")

    return dest
