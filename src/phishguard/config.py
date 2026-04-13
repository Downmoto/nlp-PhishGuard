"""Central configuration loader."""
from __future__ import annotations

import os
from pathlib import Path

import yaml


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config.yaml relative to the project root.

    Parameters
    ----------
    path:
        Explicit path to a YAML config file.  If *None* the function walks up
        from this file's location until it finds ``config.yaml``.
    """
    if path is not None:
        cfg_path = Path(path)
    else:
        # Walk up until we find config.yaml (works whether called from src/ or root)
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "config.yaml"
            if candidate.exists():
                cfg_path = candidate
                break
        else:
            raise FileNotFoundError("config.yaml not found in any parent directory.")

    with cfg_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    # Resolve relative paths against the directory that holds config.yaml
    root = cfg_path.parent
    for key in ("raw_dir", "processed_dir", "data_dir", "output_dir",
                "best_checkpoint_dir", "figures_dir", "reports_dir"):
        section = _find_section(config, key)
        if section is not None:
            section[key] = str(root / section[key])

    return config


def _find_section(config: dict, key: str) -> dict | None:
    """Return the sub-dict that contains *key*, searching one level deep."""
    if key in config:
        return config
    for v in config.values():
        if isinstance(v, dict) and key in v:
            return v
    return None
