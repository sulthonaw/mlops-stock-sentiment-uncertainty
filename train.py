"""Root entrypoint script for training machine learning models with MLflow.

Fulfills LK-06 requirements:
1. Trains ML model using DVC-versioned dataset.
2. Integrates with MLflow (log_param, log_metric, log_model, artifacts).
3. Executes hyperparameter variations (minimal 3 runs).
4. Compares runs and registers the best model to the MLflow Model Registry.
5. Generates models/model_metadata.json for subsequent deployment.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import List, Dict, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from src.models.train_model import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    train_and_log_run,
)
from src.utils.logger import get_logger

logger = get_logger("train_cli")


def run_experiment_variations(
    dataset_path: Path,
    tracking_uri: str,
    experiment_name: str,
    model_registry_name: str = "StockSentimentModel",
) -> Dict[str, Any]:
    """Execute 3 distinct hyperparameter variation runs and register the best model."""
    logger.info("=================================================================")
    logger.info("[START] EXECUTING 3 MLFLOW EXPERIMENT RUNS (HYPERPARAMETER TUNING)")
    logger.info("=================================================================")

    variations = [
        {
            "run_name": "run1_logreg_baseline_unigram",
            "classifier_type": "logistic_regression",
            "c_param": 1.0,
            "ngram_range": (1, 1),
            "max_features": 5000,
            "sublinear_tf": False,
            "tags": {"variation_stage": "baseline", "experiment_variant": "unigram_c1.0"},
        },
        {
            "run_name": "run2_logreg_tuned_bigram_sublinear",
            "classifier_type": "logistic_regression",
            "c_param": 2.5,
            "ngram_range": (1, 2),
            "max_features": 8000,
            "sublinear_tf": True,
            "tags": {"variation_stage": "tuned_nlp", "experiment_variant": "bigram_sublinear_c2.5"},
        },
        {
            "run_name": "run3_linearsvc_margin_maximizer",
            "classifier_type": "linear_svc",
            "c_param": 1.0,
            "ngram_range": (1, 2),
            "max_features": 8000,
            "sublinear_tf": True,
            "tags": {"variation_stage": "alternative_classifier", "experiment_variant": "linear_svc_bigram"},
        },
    ]

    run_results: List[Dict[str, Any]] = []

    for idx, var in enumerate(variations, 1):
        logger.info(f"\n--- [VARIATION {idx}/3]: {var['run_name']} ---")
        res = train_and_log_run(
            dataset_path=dataset_path,
            classifier_type=var["classifier_type"],
            c_param=var["c_param"],
            ngram_range=var["ngram_range"],
            max_features=var["max_features"],
            sublinear_tf=var["sublinear_tf"],
            experiment_name=experiment_name,
            run_name=var["run_name"],
            tracking_uri=tracking_uri,
            tags=var["tags"],
        )
        run_results.append(res)

    # 4. Compare Runs and Identify Best Model
    logger.info("\n=================================================================")
    logger.info("[SUMMARY] EXPERIMENT COMPARISON SUMMARY (3 RUNS)")
    logger.info("=================================================================")
    logger.info(f"{'Run Name':<35} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<10}")
    logger.info("-" * 75)

    best_run = None
    highest_f1 = -1.0

    for res in run_results:
        m = res["metrics"]
        name = res["run_name"]
        acc = m["val_accuracy"]
        macro_f1 = m["val_f1_macro"]
        weighted_f1 = m["val_f1_weighted"]
        logger.info(f"{name:<35} | {acc:.4f}     | {macro_f1:.4f}     | {weighted_f1:.4f}")

        if macro_f1 > highest_f1:
            highest_f1 = macro_f1
            best_run = res

    logger.info("-" * 75)
    logger.info(
        f"[BEST MODEL] Best Performing Run: '{best_run['run_name']}' with Val Macro F1 = {highest_f1:.4f}"
    )

    # 5. Register Best Model to MLflow Model Registry
    logger.info(f"\n[REGISTRY] Registering best model to MLflow Model Registry as '{model_registry_name}'...")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{best_run['run_id']}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_registry_name,
        tags={
            "selection_criterion": "highest_val_f1_macro",
            "source_run_id": best_run["run_id"],
            "registered_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    client = MlflowClient(tracking_uri=tracking_uri)
    client.set_model_version_tag(
        name=model_registry_name,
        version=registered_model.version,
        key="status",
        value="candidate_production",
    )
    logger.info(
        f"[SUCCESS] Model registered successfully! Name: '{model_registry_name}', Version: {registered_model.version}"
    )

    # 6. Save Model Metadata (LK-06 Luaran 3)
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = models_dir / "model_metadata.json"

    metadata = {
        "model_registry_name": model_registry_name,
        "model_version": registered_model.version,
        "best_run_id": best_run["run_id"],
        "best_run_name": best_run["run_name"],
        "selection_metric": "val_f1_macro",
        "best_metric_value": highest_f1,
        "metrics": best_run["metrics"],
        "parameters": best_run["params"],
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "deployment_recommendation": {
            "target_service": "FastAPI Sentiment Inference Service",
            "batch_infer_support": True,
            "latency": "low (<15ms)",
        },
    }

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"[METADATA] Model deployment metadata successfully written to: {metadata_file}")
    logger.info("\n=================================================================")
    logger.info("[COMPLETE] EXPERIMENT & REGISTRY COMPLETED SUCCESSFULLY!")
    logger.info("To view visual dashboard and compare runs, start MLflow UI:")
    logger.info(f"    mlflow ui --backend-store-uri {tracking_uri} --port 5000")
    logger.info("=================================================================\n")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="MLflow Stock Sentiment Model Training CLI"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/IDSMSA.csv",
        help="Path to dataset CSV (default: data/raw/IDSMSA.csv)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help="MLflow tracking URI (default: sqlite:///mlruns.db)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=DEFAULT_EXPERIMENT_NAME,
        help="MLflow experiment name (default: stock-sentiment-analysis)",
    )
    parser.add_argument(
        "--run-variations",
        action="store_true",
        default=False,
        help="Execute 3 hyperparameter variations and register best model automatically",
    )
    parser.add_argument(
        "--classifier",
        type=str,
        default="logistic_regression",
        choices=["logistic_regression", "linear_svc", "random_forest"],
        help="Classifier type for single run",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.0,
        help="Regularization parameter C",
    )
    parser.add_argument(
        "--ngram-max",
        type=int,
        default=1,
        help="Max n-gram for TF-IDF (1=unigram, 2=bigram)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=5000,
        help="Max vocabulary features",
    )
    parser.add_argument(
        "--sublinear-tf",
        action="store_true",
        default=False,
        help="Apply sublinear TF scaling",
    )
    parser.add_argument(
        "--register-model",
        type=str,
        default=None,
        help="Model Registry name if registering single run",
    )

    args = parser.parse_args()
    dataset_path = PROJECT_ROOT / args.dataset

    # If run_variations is specified or no specific tuning overrides were provided
    if args.run_variations:
        run_experiment_variations(
            dataset_path=dataset_path,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
        )
    else:
        logger.info("Running single model training run...")
        train_and_log_run(
            dataset_path=dataset_path,
            classifier_type=args.classifier,
            c_param=args.c,
            ngram_range=(1, args.ngram_max),
            max_features=args.max_features,
            sublinear_tf=args.sublinear_tf,
            experiment_name=args.experiment,
            tracking_uri=args.tracking_uri,
            registered_model_name=args.register_model,
        )


if __name__ == "__main__":
    main()
