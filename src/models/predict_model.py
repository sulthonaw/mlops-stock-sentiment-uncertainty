"""Module for programmatic inference using MLflow Model Registry models.

Fulfills LK-07 Requirement 5:
Verifies that models in the 'Production' (or 'Staging') stage can be loaded programmatically
using mlflow.pyfunc.load_model for real-world serving readiness.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Union

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

from src.preprocess import clean_text_idsmsa
from src.utils.logger import get_logger

logger = get_logger("predict_model")

DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"
DEFAULT_MODEL_NAME = "StockSentimentClassifier"
DEFAULT_STAGE = "Production"


def load_production_model(
    model_name: str = DEFAULT_MODEL_NAME,
    stage: str = DEFAULT_STAGE,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    version: Optional[int] = None,
) -> Any:
    """Programmatically load model from MLflow Model Registry via mlflow.pyfunc.

    Args:
        model_name: Name of registered model in MLflow.
        stage: Stage to load ('Production', 'Staging', etc.). Ignored if version is specified.
        tracking_uri: Tracking URI of MLflow backend store.
        version: Specific integer version to load (if specified).

    Returns:
        Loaded PyFunc model ready for inference.
    """
    mlflow.set_tracking_uri(tracking_uri)

    if version is not None:
        model_uri = f"models:/{model_name}/{version}"
    else:
        # Load by stage or modern alias
        model_uri = f"models:/{model_name}/{stage}"

    logger.info(f"[LOAD] Loading model from URI: '{model_uri}' (Tracking URI: {tracking_uri})...")
    start_t = time.perf_counter()

    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        logger.warning(
            f"Failed to load directly via '{model_uri}' ({exc}). Attempting alias fallback..."
        )
        try:
            # Fallback to alias if supported
            alias = stage.lower()
            alias_uri = f"models:/{model_name}@{alias}"
            logger.info(f"Trying alias URI: '{alias_uri}'...")
            model = mlflow.pyfunc.load_model(alias_uri)
        except Exception:
            # Fallback to query client for current production version
            client = MlflowClient(tracking_uri=tracking_uri)
            versions = client.search_model_versions(f"name='{model_name}'")
            prod_versions = [v for v in versions if v.current_stage.lower() == stage.lower()]
            if not prod_versions:
                raise RuntimeError(
                    f"No model found with name '{model_name}' and stage '{stage}'."
                )
            prod_v = sorted(prod_versions, key=lambda x: int(x.version))[-1]
            version_uri = f"models:/{model_name}/{prod_v.version}"
            logger.info(f"Fallback to resolved version URI: '{version_uri}'...")
            model = mlflow.pyfunc.load_model(version_uri)

    load_duration_ms = (time.perf_counter() - start_t) * 1000
    logger.info(f"[READY] Model successfully loaded in {load_duration_ms:.2f} ms!")
    return model


def predict_sentiment(
    texts: Union[str, List[str]],
    model: Optional[Any] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    stage: str = DEFAULT_STAGE,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    clean_text: bool = True,
) -> List[Dict[str, Any]]:
    """Execute inference for single or batch Indonesian stock sentiment texts.

    Args:
        texts: Single string or list of text sentences.
        model: Pre-loaded PyFunc model. If None, loaded automatically via load_production_model().
        model_name: Registered model name.
        stage: Model stage to load if model is None.
        tracking_uri: MLflow tracking URI.
        clean_text: Whether to apply ID-SMSA standard preprocessing.

    Returns:
        List of dicts containing original text, cleaned text, predicted sentiment, and latency.
    """
    if isinstance(texts, str):
        text_list = [texts]
    else:
        text_list = list(texts)

    if model is None:
        model = load_production_model(model_name=model_name, stage=stage, tracking_uri=tracking_uri)

    # Preprocessing
    processed_texts = []
    for t in text_list:
        if clean_text:
            cleaned = clean_text_idsmsa(str(t))
            # Fallback if cleaning yields empty string
            processed_texts.append(cleaned if len(cleaned.strip()) > 0 else str(t))
        else:
            processed_texts.append(str(t))

    # Inference execution & timing
    t0 = time.perf_counter()
    input_data = pd.Series(processed_texts)
    predictions = model.predict(input_data)
    total_latency_ms = (time.perf_counter() - t0) * 1000
    per_item_latency_ms = total_latency_ms / max(1, len(text_list))

    results: List[Dict[str, Any]] = []
    for orig, proc, pred in zip(text_list, processed_texts, predictions):
        results.append({
            "original_text": orig,
            "processed_text": proc,
            "predicted_sentiment": str(pred).capitalize(),
            "per_item_latency_ms": round(per_item_latency_ms, 2),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Programmatic Inference for Stock Sentiment Model")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Registered model name",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=DEFAULT_STAGE,
        choices=["Production", "Staging", "None"],
        help="Model stage to load",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=None,
        help="Specific version number (optional)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text snippet to classify",
    )

    args = parser.parse_args()

    sample_texts = [
        args.text
    ] if args.text else [
        "BBCA berhasil mencetak rekor laba bersih tertinggi sepanjang sejarah dan berencana bagi dividen interim jumbo.",
        "Saham TLKM anjlok tajam menembus support psikologis setelah sentimen negatif asing dan perang tarif telekomunikasi.",
        "IHSG ditutup stagnan cenderung konsolidasi tipis hari ini menanti rilis neraca perdagangan kuartal III.",
    ]

    logger.info("Executing sample inference test...")
    model = load_production_model(
        model_name=args.model_name,
        stage=args.stage,
        tracking_uri=args.tracking_uri,
        version=args.version,
    )

    results = predict_sentiment(texts=sample_texts, model=model)

    print("\n" + "=" * 80)
    print("INFERENCE VERIFICATION RESULTS")
    print("=" * 80)
    for idx, res in enumerate(results, 1):
        print(f"\n[Case {idx}]")
        print(f"  Input Text : {res['original_text']}")
        print(f"  Cleaned    : {res['processed_text']}")
        print(f"  Sentiment  : {res['predicted_sentiment']}")
        print(f"  Latency    : {res['per_item_latency_ms']} ms")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()

