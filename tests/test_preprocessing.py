"""Unit tests for text preprocessing and cleaning routines."""

import pytest
import pandas as pd
from src.preprocess import clean_text_idsmsa


def test_clean_text_url_replacement():
    """Verify that HTTP and HTTPS URLs are converted to [URL] token."""
    raw = "Cek laporan keuangan emiten di https://idx.co.id/financial atau www.bareksa.com"
    expected = "Cek laporan keuangan emiten di [URL] atau [URL]"
    assert clean_text_idsmsa(raw) == expected


def test_clean_text_username_replacement():
    """Verify that user account mentions are converted to [USERNAME] token."""
    raw = "Terima kasih atas analisanya @investor_muda dan @stockbit_official"
    expected = "Terima kasih atas analisanya [USERNAME] dan [USERNAME]"
    assert clean_text_idsmsa(raw) == expected


def test_clean_text_hashtag_replacement():
    """Verify that hashtags are converted to [HASHTAG] token."""
    raw = "Saham #BBCA dan #BBRI melesat pagi ini #IHSG"
    expected = "Saham [HASHTAG] dan [HASHTAG] melesat pagi ini [HASHTAG]"
    assert clean_text_idsmsa(raw) == expected


def test_clean_text_whitespace_normalization():
    """Verify that duplicate whitespace, tabs, and newlines are collapsed."""
    raw = "  Saham    BMRI \n\n  berhasil \t rebound kuat.   "
    expected = "Saham BMRI berhasil rebound kuat."
    assert clean_text_idsmsa(raw) == expected


def test_clean_text_null_and_empty():
    """Verify that None, pd.NA, and empty strings return an empty string safely."""
    assert clean_text_idsmsa(None) == ""
    assert clean_text_idsmsa("") == ""
    assert clean_text_idsmsa(pd.NA) == ""


def test_clean_text_combined_idsmsa_sample():
    """Verify an end-to-end composite stock tweet text sample."""
    raw = "BREAKING: @analis_saham memprediksi #TLKM rebound ke 3200! Info: https://t.co/abc123xyz"
    cleaned = clean_text_idsmsa(raw)
    assert "[USERNAME]" in cleaned
    assert "[HASHTAG]" in cleaned
    assert "[URL]" in cleaned
    assert "https://" not in cleaned
    assert "@" not in cleaned
    assert "#" not in cleaned
