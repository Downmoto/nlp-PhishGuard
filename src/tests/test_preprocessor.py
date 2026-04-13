"""Unit tests for phishguard.data.preprocessor."""
import pytest

from phishguard.data.preprocessor import (
    normalise_urls,
    normalise_whitespace,
    preprocess_text,
    remove_pii,
    strip_html,
)


class TestStripHtml:
    def test_removes_tags(self):
        assert "<b>" not in strip_html("<b>Hello</b>")

    def test_preserves_text(self):
        result = strip_html("<p>Keep this text</p>")
        assert "Keep this text" in result

    def test_non_string_returns_empty(self):
        assert strip_html(None) == ""  # type: ignore[arg-type]


class TestRemovePii:
    def test_masks_email(self):
        result = remove_pii("Contact foo@bar.com for details")
        assert "foo@bar.com" not in result
        assert "[EMAIL]" in result

    def test_leaves_non_email_unchanged(self):
        result = remove_pii("No emails here")
        assert result == "No emails here"


class TestNormaliseUrls:
    def test_replaces_http_url(self):
        result = normalise_urls("Visit http://example.com/page?q=1 now")
        assert "http://example.com" not in result
        assert "[URL]" in result

    def test_replaces_www_url(self):
        result = normalise_urls("Go to www.phish.io today")
        assert "[URL]" in result


class TestNormaliseWhitespace:
    def test_collapses_spaces(self):
        assert normalise_whitespace("a  b   c") == "a b c"

    def test_strips_leading_trailing(self):
        assert normalise_whitespace("  hello  ") == "hello"

    def test_newlines_collapsed(self):
        assert normalise_whitespace("a\nb\tc") == "a b c"


class TestPreprocessText:
    def test_full_pipeline(self):
        raw = "<p>Dear user@test.com, click http://evil.com to win!</p>"
        result = preprocess_text(raw)
        assert "<p>" not in result
        assert "user@test.com" not in result
        assert "http://evil.com" not in result
        assert "[EMAIL]" in result
        assert "[URL]" in result
