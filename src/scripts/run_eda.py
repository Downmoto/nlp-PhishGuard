#!/usr/bin/env python
"""Run EDA on the training split and save figures to reports/figures/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from phishguard.config import load_config
from phishguard.eda.eda import run_all


def main() -> None:
    config = load_config()
    processed_dir = config["data"]["processed_dir"]
    figures_dir = config["output"]["figures_dir"]

    train_path = Path(processed_dir) / "train.parquet"
    if not train_path.exists():
        print(
            f"[run_eda] Processed data not found at {train_path}.\n"
            "Run scripts/download_data.py and then the data loading step first."
        )
        sys.exit(1)

    df = pd.read_parquet(train_path)
    run_all(df, figures_dir)


if __name__ == "__main__":
    main()
