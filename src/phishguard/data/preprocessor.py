"""Text pre-processing: PII removal, HTML stripping, URL normalisation."""
from __future__ import annotations

import re

import pandas as pd

# Attempt a fast import; bs4 is only required when HTML stripping is used.
try:
    from bs4 import BeautifulSoup

    _BS4_AVAILABLE = True
except ImportError:  # pragma: no cover
    _BS4_AVAILABLE = False

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
_RE_EMAIL = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.IGNORECASE)
_RE_PHONE = re.compile(
    r"(\+?\d[\d\s\-().]{7,}\d)",
    re.IGNORECASE,
)
_RE_URL = re.compile(
    r"(https?://\S+|www\.\S+)",
    re.IGNORECASE,
)
_RE_WHITESPACE = re.compile(r"\s+")


def strip_html(text: str) -> str:
    """Remove HTML tags.  Falls back to a regex approach if bs4 is unavailable."""
    if not isinstance(text, str):
        return ""
    if _BS4_AVAILABLE:
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text(separator=" ")
    # Fallback: simple tag removal
    return re.sub(r"<[^>]+>", " ", text)


def remove_pii(text: str) -> str:
    """Replace email addresses and phone-number patterns with placeholder tokens."""
    text = _RE_EMAIL.sub("[EMAIL]", text)
    text = _RE_PHONE.sub("[PHONE]", text)
    return text


def normalise_urls(text: str) -> str:
    """Replace URLs with the ``[URL]`` token."""
    return _RE_URL.sub("[URL]", text)


def normalise_whitespace(text: str) -> str:
    """Collapse consecutive whitespace characters into a single space."""
    return _RE_WHITESPACE.sub(" ", text).strip()


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline for a single email body string.

    Steps (in order):
    1. HTML stripping
    2. URL normalisation
    3. PII removal
    4. Whitespace normalisation
    """
    if not isinstance(text, str):
        return ""
    text = strip_html(text)
    text = normalise_urls(text)
    text = remove_pii(text)
    text = normalise_whitespace(text)
    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Apply :func:`preprocess_text` to every row of *text_col* in *df*.

    Returns a copy of the DataFrame with the processed column.
    """
    out = df.copy()
    out[text_col] = out[text_col].apply(preprocess_text)
    return out
