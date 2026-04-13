"""Load raw CSV data, normalise labels, split, and save as Parquet."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Mapping of raw label values → integer (1 = phishing/spam, 0 = legitimate)
_LABEL_MAP: dict[str, int] = {
    # numeric strings
    "1": 1, "0": 0,
    # text variants (phishing_email.csv uses these)
    "spam": 1, "phishing": 1, "phish": 1,
    "ham": 0, "legitimate": 0, "legit": 0, "safe": 0,
}


def _normalise_label(series: pd.Series) -> pd.Series:
    """Convert label column to integer 0/1."""
    if pd.api.types.is_integer_dtype(series):
        return series.astype(int)
    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(_LABEL_MAP)
    if mapped.isna().any():
        unknown = lowered[mapped.isna()].unique().tolist()
        raise ValueError(f"Unknown label values: {unknown}")
    return mapped.astype(int)


def load_primary(raw_dir: str | os.PathLike, filename: str = "phishing_email.csv") -> pd.DataFrame:
    """Read the primary phishing dataset and normalise it.

    Expected columns (after normalisation): ``text``, ``label``.

    The raw CSV may expose the email body under different column names;
    this function attempts to resolve the correct one.
    """
    path = Path(raw_dir) / filename
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()

    # Resolve text column
    text_candidates = ["text_combined", "email_text", "body", "text", "message", "content", "email"]
    text_col = next((c for c in text_candidates if c in df.columns), None)
    if text_col is None:
        raise KeyError(
            f"Cannot find a text column in {filename}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Resolve label column
    label_candidates = ["email_type", "label", "class", "type", "spam", "phishing"]
    label_col = next((c for c in label_candidates if c in df.columns), None)
    if label_col is None:
        raise KeyError(
            f"Cannot find a label column in {filename}. "
            f"Available columns: {df.columns.tolist()}"
        )

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["label"] = _normalise_label(df["label"])
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]
    return df.reset_index(drop=True)


def verify_balance(df: pd.DataFrame) -> None:
    """Print label distribution for quick sanity check."""
    counts = df["label"].value_counts().sort_index()
    total = len(df)
    print("[loader] Label distribution:")
    for lbl, cnt in counts.items():
        name = "phishing" if lbl == 1 else "legitimate"
        print(f"  {lbl} ({name}): {cnt:,}  ({cnt / total:.1%})")


def split_and_save(
    df: pd.DataFrame,
    processed_dir: str | os.PathLike,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Stratified 70/15/15 split saved as Parquet files.

    Returns
    -------
    dict
        Keys ``"train"``, ``"val"``, ``"test"`` each mapping to a DataFrame.
    """
    out = Path(processed_dir)
    out.mkdir(parents=True, exist_ok=True)

    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, "train + val ratios must sum to less than 1."

    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), stratify=df["label"], random_state=random_seed
    )
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - relative_val), stratify=temp_df["label"], random_state=random_seed
    )

    splits: dict[str, pd.DataFrame] = {"train": train_df, "val": val_df, "test": test_df}
    for name, split_df in splits.items():
        dest = out / f"{name}.parquet"
        split_df.reset_index(drop=True).to_parquet(dest, index=False)
        print(f"[loader] Saved {name} ({len(split_df):,} rows) → {dest}")

    return splits
