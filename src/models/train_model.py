"""Module for model training and MLflow experiment tracking.

Implements text classification pipeline for Indonesian Stock Sentiment Analysis
integrated with MLflow for parameter logging, metric tracking, artifact storage,
and model registration.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import yaml

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import clean_text_idsmsa
from src.utils.logger import get_logger

logger = get_logger("model_training")

DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"
DEFAULT_EXPERIMENT_NAME = "stock-sentiment-analysis"


def load_dvc_dataset_metadata(dvc_file_path: Path) -> Dict[str, Any]:
    """Read metadata (md5 hash, size) from a DVC pointer file."""
    if not dvc_file_path.exists():
        return {}
    try:
        with open(dvc_file_path, "r", encoding="utf-8") as f:
            dvc_data = yaml.safe_load(f)
            outs = dvc_data.get("outs", [])
            if outs:
                return {
                    "dvc_md5": outs[0].get("md5"),
                    "dvc_size_bytes": outs[0].get("size"),
                    "dvc_pointer_file": str(dvc_file_path),
                }
    except Exception as exc:
        logger.warning(f"Could not parse DVC file {dvc_file_path}: {exc}")
    return {}


def load_and_prepare_data(
    dataset_path: Path,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    """Load dataset, clean text using ID-SMSA standards, and split into train/test sets."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    logger.info(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)

    # Validate required columns
    if "Sentence" not in df.columns or "Sentiment" not in df.columns:
        raise ValueError("Dataset must contain 'Sentence' and 'Sentiment' columns.")

    initial_len = len(df)
    # Drop records with missing sentence or sentiment
    df = df.dropna(subset=["Sentence", "Sentiment"]).copy()
    df["Sentence"] = df["Sentence"].astype(str)
    df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.capitalize()

    # Apply standardization text cleaning
    df["cleaned_sentence"] = df["Sentence"].apply(clean_text_idsmsa)
    # Filter empty texts
    df = df[df["cleaned_sentence"].str.len() > 0].copy()

    logger.info(
        f"Preprocessed {len(df)} valid records (dropped {initial_len - len(df)} null/empty)."
    )
    logger.info(f"Sentiment distribution:\n{df['Sentiment'].value_counts().to_dict()}")

    X = df["cleaned_sentence"]
    y = df["Sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    metadata = {
        "total_records": len(df),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "class_distribution": df["Sentiment"].value_counts().to_dict(),
        "classes": sorted(list(y.unique())),
    }

    return X_train, X_test, y_train, y_test, metadata


def build_pipeline(
    classifier_type: str = "logistic_regression",
    c_param: float = 1.0,
    ngram_range: Tuple[int, int] = (1, 1),
    max_features: int = 5000,
    sublinear_tf: bool = False,
    random_state: int = 42,
) -> Pipeline:
    """Build Scikit-Learn TF-IDF + Classifier pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        min_df=2,
    )

    if classifier_type.lower() == "logistic_regression":
        classifier = LogisticRegression(
            C=c_param,
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
            solver="lbfgs",
        )
    elif classifier_type.lower() == "linear_svc":
        classifier = LinearSVC(
            C=c_param,
            random_state=random_state,
            class_weight="balanced",
            max_iter=2000,
        )
    elif classifier_type.lower() == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=random_state,
            class_weight="balanced",
        )
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def train_and_log_run(
    dataset_path: Path = PROJECT_ROOT / "data" / "raw" / "IDSMSA.csv",
    classifier_type: str = "logistic_regression",
    c_param: float = 1.0,
    ngram_range: Tuple[int, int] = (1, 1),
    max_features: int = 5000,
    sublinear_tf: bool = False,
    random_state: int = 42,
    test_size: float = 0.2,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: Optional[str] = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    registered_model_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Train pipeline and log full experiment run to MLflow."""
    # 1. Setup MLflow Tracking
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # 2. Prepare Data
    X_train, X_test, y_train, y_test, data_meta = load_and_prepare_data(
        dataset_path=dataset_path,
        test_size=test_size,
        random_state=random_state,
    )

    # 3. Check DVC metadata for reproducibility tracking
    dvc_file = dataset_path.with_suffix(".csv.dvc")
    dvc_meta = load_dvc_dataset_metadata(dvc_file)

    # 4. Construct Pipeline
    pipeline = build_pipeline(
        classifier_type=classifier_type,
        c_param=c_param,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        random_state=random_state,
    )

    auto_run_name = run_name or (
        f"{classifier_type}_c{c_param}_ngram{ngram_range[0]}-{ngram_range[1]}"
    )

    logger.info(f"Starting MLflow Run: {auto_run_name} in experiment '{experiment_name}'")

    with mlflow.start_run(run_name=auto_run_name) as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Set Tags
        default_tags = {
            "mlops_phase": "experiment_tracking",
            "task": "stock_sentiment_classification",
            "framework": "scikit-learn",
            "model_type": classifier_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if tags:
            default_tags.update(tags)
        mlflow.set_tags(default_tags)

        # Log Parameters
        params = {
            "classifier_type": classifier_type,
            "c_regularization": c_param,
            "ngram_min": ngram_range[0],
            "ngram_max": ngram_range[1],
            "max_features": max_features,
            "sublinear_tf": sublinear_tf,
            "random_state": random_state,
            "test_size": test_size,
            "dataset_path": str(dataset_path.relative_to(PROJECT_ROOT)),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }
        if dvc_meta:
            params.update(dvc_meta)
        mlflow.log_params(params)

        # Train Model
        logger.info(f"Fitting {classifier_type} pipeline on {len(X_train)} samples...")
        pipeline.fit(X_train, y_train)

        # Predictions & Evaluations
        y_pred = pipeline.predict(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1_macro = float(f1_score(y_test, y_pred, average="macro"))
        f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
        precision_macro = float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        )
        recall_macro = float(
            recall_score(y_test, y_pred, average="macro", zero_division=0)
        )

        metrics = {
            "val_accuracy": acc,
            "val_f1_macro": f1_macro,
            "val_f1_weighted": f1_weighted,
            "val_precision_macro": precision_macro,
            "val_recall_macro": recall_macro,
        }
        mlflow.log_metrics(metrics)
        logger.info(
            f"Evaluation Metrics -> Accuracy: {acc:.4f} | Macro F1: {f1_macro:.4f} | Weighted F1: {f1_weighted:.4f}"
        )

        # Generate & Log Confusion Matrix Heatmap Artifact
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            labels = sorted(list(np.unique(y_test)))
            cm = confusion_matrix(y_test, y_pred, labels=labels)
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=labels,
                yticklabels=labels,
            )
            plt.title(f"Confusion Matrix: {auto_run_name}\nMacro F1: {f1_macro:.4f}")
            plt.xlabel("Predicted Sentiment")
            plt.ylabel("Actual Sentiment")
            plt.tight_layout()

            cm_path = tmp_path / "confusion_matrix.png"
            plt.savefig(cm_path, dpi=200)
            plt.close()
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation")

            # Classification Report JSON
            report_dict = classification_report(
                y_test, y_pred, labels=labels, output_dict=True, zero_division=0
            )
            report_path = tmp_path / "classification_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2)
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")

        # Model Logging & Signature
        sample_input = np.array(
            ["Saham BBCA berhasil rebound kuat dan mencatatkan laba bersih rekor"]
        )
        sample_output = pipeline.predict(sample_input)
        signature = infer_signature(sample_input, sample_output)

        try:
            # MLflow 3.x accepts name="model"
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                signature=signature,
                input_example=sample_input,
                registered_model_name=registered_model_name,
                serialization_format="cloudpickle",
            )
        except TypeError:
            # Fallback for older MLflow accepting artifact_path
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="model",
                signature=signature,
                input_example=sample_input,
                registered_model_name=registered_model_name,
                serialization_format="cloudpickle",
            )

        logger.info(f"Model successfully logged to MLflow Run: {run_id}")

        return {
            "run_id": run_id,
            "run_name": auto_run_name,
            "metrics": metrics,
            "params": params,
            "pipeline": pipeline,
            "experiment_name": experiment_name,
        }
