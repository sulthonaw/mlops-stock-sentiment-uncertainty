"""Unit tests for dataset schema, integrity, and DVC tracking configuration."""

from pathlib import Path
import yaml
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DVC_POINTER_FILE = DATA_DIR / "IDSMSA.csv.dvc"
DATASET_CSV_FILE = DATA_DIR / "IDSMSA.csv"


def test_dvc_pointer_file_exists():
    """Verify that the DVC pointer file exists in the repository."""
    assert DVC_POINTER_FILE.exists(), f"DVC pointer file missing at {DVC_POINTER_FILE}"


def test_dvc_pointer_structure():
    """Verify that the DVC pointer file contains valid metadata (md5, size, path)."""
    with open(DVC_POINTER_FILE, "r", encoding="utf-8") as f:
        dvc_data = yaml.safe_load(f)

    assert "outs" in dvc_data, "DVC pointer file missing 'outs' key"
    outs = dvc_data["outs"]
    assert len(outs) > 0, "DVC 'outs' list is empty"
    
    first_out = outs[0]
    assert "md5" in first_out, "DVC metadata missing 'md5' hash"
    assert "path" in first_out, "DVC metadata missing 'path'"
    assert first_out["path"] == "IDSMSA.csv"
    assert len(first_out["md5"]) == 32, "Invalid MD5 hash format in DVC file"


def test_dataset_columns_and_labels():
    """Verify that the dataset conforms to required schema when the file is present."""
    if not DATASET_CSV_FILE.exists():
        pytest.skip(f"Dataset {DATASET_CSV_FILE} not pulled yet; skipping file content check.")

    df = pd.read_csv(DATASET_CSV_FILE)
    assert "Sentence" in df.columns, "Missing 'Sentence' column in dataset"
    assert "Sentiment" in df.columns, "Missing 'Sentiment' column in dataset"

    # Verify sentiment classes
    valid_labels = {"positive", "neutral", "negative"}
    unique_labels = set(df["Sentiment"].dropna().astype(str).str.strip().str.lower().unique())
    assert unique_labels.issubset(valid_labels), f"Unexpected sentiment labels found: {unique_labels}"

    # Verify non-empty
    assert len(df) > 0, "Dataset has 0 rows"
