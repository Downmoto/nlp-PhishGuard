#!/usr/bin/env python
"""Download the PhishGuard training datasets from Kaggle."""
import sys
from pathlib import Path

# Make the src package importable when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phishguard.config import load_config
from phishguard.data.downloader import download_dataset


def main() -> None:
    config = load_config()
    raw_dir = config["data"]["raw_dir"]
    primary_slug = config["data"]["primary_dataset"]

    download_dataset(primary_slug, raw_dir)


if __name__ == "__main__":
    main()
