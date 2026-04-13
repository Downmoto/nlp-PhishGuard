"""Exploratory Data Analysis for the PhishGuard dataset."""
from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from wordcloud import WordCloud

    _WORDCLOUD_AVAILABLE = True
except ImportError:
    _WORDCLOUD_AVAILABLE = False

try:
    import nltk
    from nltk.corpus import stopwords

    try:
        _STOPWORDS = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        _STOPWORDS = set(stopwords.words("english"))
except ImportError:
    _STOPWORDS = set()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_LABEL_NAMES = {0: "Legitimate", 1: "Phishing"}


def _ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tokenise(text: str) -> list[str]:
    tokens = re.findall(r"\b[a-z]{3,}\b", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Individual analyses
# ---------------------------------------------------------------------------

def plot_class_distribution(df: pd.DataFrame, figures_dir: str | os.PathLike) -> None:
    """Bar chart of label counts."""
    out = _ensure_dir(figures_dir)
    counts = df["label"].map(_LABEL_NAMES).value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="black")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.bar_label(ax.containers[0], fmt="%d")
    plt.tight_layout()
    fig.savefig(out / "class_distribution.png", dpi=150)
    plt.close(fig)
    print("[eda] Saved class_distribution.png")


def plot_length_distribution(df: pd.DataFrame, figures_dir: str | os.PathLike) -> None:
    """Body word-count distributions per class."""
    out = _ensure_dir(figures_dir)
    df = df.copy()
    df["word_count"] = df["text"].str.split().str.len()
    df["class"] = df["label"].map(_LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group in df.groupby("class"):
        wc = group["word_count"].clip(upper=2000)
        ax.hist(wc, bins=60, alpha=0.6, label=label)
    ax.set_title("Email Length Distribution (word count, capped at 2000)")
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out / "length_distribution.png", dpi=150)
    plt.close(fig)
    print("[eda] Saved length_distribution.png")


def plot_top_words(
    df: pd.DataFrame,
    figures_dir: str | os.PathLike,
    top_n: int = 20,
) -> None:
    """Horizontal bar charts of top-N words per class."""
    out = _ensure_dir(figures_dir)
    for label_id, label_name in _LABEL_NAMES.items():
        texts = df.loc[df["label"] == label_id, "text"].tolist()
        all_tokens: list[str] = []
        for t in texts:
            all_tokens.extend(_tokenise(t))
        common = Counter(all_tokens).most_common(top_n)
        words, freqs = zip(*common) if common else ([], [])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(list(words)[::-1], list(freqs)[::-1], color="steelblue" if label_id == 0 else "tomato")
        ax.set_title(f"Top {top_n} Words — {label_name}")
        ax.set_xlabel("Frequency")
        plt.tight_layout()
        fig.savefig(out / f"top_words_{label_name.lower()}.png", dpi=150)
        plt.close(fig)
        print(f"[eda] Saved top_words_{label_name.lower()}.png")


def plot_wordclouds(df: pd.DataFrame, figures_dir: str | os.PathLike) -> None:
    """Word clouds for phishing and legitimate emails."""
    if not _WORDCLOUD_AVAILABLE:
        print("[eda] wordcloud package not installed — skipping word cloud plots.")
        return
    out = _ensure_dir(figures_dir)
    for label_id, label_name in _LABEL_NAMES.items():
        corpus = " ".join(df.loc[df["label"] == label_id, "text"].tolist())
        tokens = _tokenise(corpus)
        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(tokens))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {label_name}")
        plt.tight_layout()
        fig.savefig(out / f"wordcloud_{label_name.lower()}.png", dpi=150)
        plt.close(fig)
        print(f"[eda] Saved wordcloud_{label_name.lower()}.png")


def plot_url_presence(df: pd.DataFrame, figures_dir: str | os.PathLike) -> None:
    """URL presence rate per class."""
    out = _ensure_dir(figures_dir)
    url_re = re.compile(r"https?://|www\.", re.IGNORECASE)
    df = df.copy()
    df["has_url"] = df["text"].str.contains(url_re)
    df["class"] = df["label"].map(_LABEL_NAMES)

    rate = df.groupby("class")["has_url"].mean() * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    rate.plot(kind="bar", ax=ax, color=["steelblue", "tomato"], edgecolor="black")
    ax.set_title("URL Presence Rate per Class (%)")
    ax.set_ylabel("% Emails containing a URL")
    ax.set_xlabel("Class")
    ax.bar_label(ax.containers[0], fmt="%.1f%%")
    plt.xticks(rotation=0)
    plt.tight_layout()
    fig.savefig(out / "url_presence_rate.png", dpi=150)
    plt.close(fig)
    print("[eda] Saved url_presence_rate.png")


def plot_source_heatmap(
    df: pd.DataFrame,
    figures_dir: str | os.PathLike,
    source_col: str = "source",
) -> None:
    """Label balance heatmap across source sub-datasets.

    Only runs if the DataFrame contains a *source* column (populated when
    loading individual CSVs rather than the combined file).
    """
    if source_col not in df.columns:
        print("[eda] No 'source' column found — skipping source heatmap.")
        return
    out = _ensure_dir(figures_dir)
    pivot = df.groupby([source_col, "label"]).size().unstack(fill_value=0)
    pivot.columns = [_LABEL_NAMES.get(c, str(c)) for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, max(4, len(pivot) // 2)))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_title("Label Balance Across Source Datasets")
    ax.set_xlabel("Class")
    ax.set_ylabel("Source")
    plt.tight_layout()
    fig.savefig(out / "source_heatmap.png", dpi=150)
    plt.close(fig)
    print("[eda] Saved source_heatmap.png")


def print_summary(df: pd.DataFrame) -> None:
    """Print key statistics to stdout."""
    print("\n========== EDA Summary ==========")
    print(f"Total emails : {len(df):,}")
    print(f"Phishing     : {(df['label'] == 1).sum():,}  ({(df['label'] == 1).mean():.1%})")
    print(f"Legitimate   : {(df['label'] == 0).sum():,}  ({(df['label'] == 0).mean():.1%})")
    wc = df["text"].str.split().str.len()
    print(f"Avg words    : {wc.mean():.1f}  (median {wc.median():.0f}, max {wc.max():,})")
    print("=================================\n")


def run_all(df: pd.DataFrame, figures_dir: str | os.PathLike) -> None:
    """Execute the full EDA suite."""
    plot_class_distribution(df, figures_dir)
    plot_length_distribution(df, figures_dir)
    plot_top_words(df, figures_dir)
    plot_wordclouds(df, figures_dir)
    plot_url_presence(df, figures_dir)
    plot_source_heatmap(df, figures_dir)
    print_summary(df)
