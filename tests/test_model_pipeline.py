"""Unit tests for model pipeline configuration, threshold definition, and inference schema."""

from pathlib import Path
import yaml
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def test_config_threshold_defined():
    """Verify that config.yaml defines the performance threshold specified since LK-01."""
    assert CONFIG_PATH.exists(), f"Configuration file missing at {CONFIG_PATH}"
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    assert "active_learning" in cfg, "Missing 'active_learning' section in config.yaml"
    al_cfg = cfg["active_learning"]
    assert "performance_f1_threshold" in al_cfg, "Missing 'performance_f1_threshold' in config.yaml"
    
    threshold = float(al_cfg["performance_f1_threshold"])
    assert 0.0 < threshold <= 1.0, f"Invalid threshold value: {threshold}"
    assert threshold == 0.75, f"Expected threshold 0.75 from LK-01, found {threshold}"


def test_lightweight_pipeline_fit_predict():
    """Verify that a minimal TF-IDF + LogisticRegression pipeline fits and predicts valid labels."""
    X_sample = [
        "Saham perbankan BBCA mencatatkan pertumbuhan laba bersih yang sangat solid dan dividen meningkat",
        "Kinerja kuartalan emiten tambang batubara anjlok tajam menembus support psikologis",
        "Pergerakan IHSG ditutup sideways cenderung mendatar menanti keputusan suku bunga acuan",
    ]
    y_sample = ["Positive", "Negative", "Neutral"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100)),
        ("classifier", LogisticRegression()),
    ])

    # Fit minimal pipeline
    pipeline.fit(X_sample, y_sample)

    # Test prediction
    preds = pipeline.predict(["BBCA berhasil cetak rekor profit", "IHSG anjlok parah"])
    assert len(preds) == 2
    for p in preds:
        assert p in ["Positive", "Negative", "Neutral"]


def test_inference_module_import_and_signature():
    """Verify that predict_model inference module is importable and has predict_sentiment function."""
    from src.models.predict_model import predict_sentiment
    assert callable(predict_sentiment)
