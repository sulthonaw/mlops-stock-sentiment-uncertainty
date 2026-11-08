"""Automated Model Evaluation Gate & Auto-Registry Script.

Fulfills LK-08 Requirements:
- Tahap 3: Model Evaluation & Validation (compares metrics against LK-01 threshold from config/config.yaml).
- Tahap 4: Auto-Registry update (promotes passing models to 'Staging' in MLflow Model Registry).
- Prevents 'silent failures' by failing the CI pipeline (exit code 1) if quality threshold is breached.
- Formats rich step summaries for GitHub Actions ($GITHUB_STEP_SUMMARY).
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Optional
import yaml

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from src.utils.logger import get_logger

logger = get_logger("evaluate_and_register")

DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"
DEFAULT_EXPERIMENT_NAME = "stock-sentiment-analysis"
DEFAULT_MODEL_NAME = "StockSentimentClassifier"
CONFIG_FILE = PROJECT_ROOT / "config" / "config.yaml"
METADATA_FILE = PROJECT_ROOT / "models" / "model_metadata.json"


def load_threshold_from_config(config_path: Path = CONFIG_FILE) -> float:
    """Read performance threshold from config/config.yaml (established since LK-01)."""
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}. Using fallback threshold 0.75.")
        return 0.75

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    threshold = cfg.get("active_learning", {}).get("performance_f1_threshold", 0.75)
    return float(threshold)


def load_latest_metadata(metadata_path: Path = METADATA_FILE) -> Dict[str, Any]:
    """Read latest model training run metadata from models/model_metadata.json."""
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at {metadata_path}. "
            "Please run 'python train.py' before evaluating."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_github_step_summary(
    status: str,
    metric_value: float,
    threshold: float,
    run_id: str,
    model_name: str,
    version: Optional[str] = None,
    stage: Optional[str] = None,
    metrics: Optional[Dict[str, float]] = None,
):
    """Write visual Markdown report table to $GITHUB_STEP_SUMMARY if running in GitHub Actions."""
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    commit_sha = os.getenv("GITHUB_SHA", "local")[:8]
    actor = os.getenv("GITHUB_ACTOR", "local-user")
    time_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "## 🤖 MLOps Automated Evaluation & Registry Gate Report",
        f"**Triggered by:** `{actor}` | **Commit:** `{commit_sha}` | **Timestamp:** `{time_str}`",
        "",
    ]

    if status == "PASSED":
        lines.extend([
            "> ### ✅ VALIDATION PASSED: Model Meets Production Quality Standards",
            f"> Metric **`val_f1_macro` ({metric_value:.4f})** is greater than or equal to threshold **`{threshold:.4f}`**.",
            "",
            "### 📦 MLflow Model Registry Status",
            f"- **Model Name:** `{model_name}`",
            f"- **Version Registered:** `v{version}`",
            f"- **Promoted Stage:** `{stage}`",
            f"- **Source MLflow Run ID:** `{run_id}`",
            "",
        ])
    else:
        lines.extend([
            "> ### ❌ VALIDATION FAILED: Performance Regression Detected (Silent Failure Prevented)",
            f"> Metric **`val_f1_macro` ({metric_value:.4f})** fell below threshold **`{threshold:.4f}`**.",
            "> The model has **NOT** been promoted to Staging to protect production stability.",
            "",
        ])

    if metrics:
        lines.extend([
            "### 📊 Detailed Validation Metrics",
            "| Metric | Value |",
            "| :--- | :--- |",
            f"| **Val Macro F1** | `{metrics.get('val_f1_macro', 0.0):.4f}` |",
            f"| **Val Accuracy** | `{metrics.get('val_accuracy', 0.0):.4f}` |",
            f"| **Val Weighted F1** | `{metrics.get('val_f1_weighted', 0.0):.4f}` |",
            f"| **Val Macro Precision** | `{metrics.get('val_precision_macro', 0.0):.4f}` |",
            f"| **Val Macro Recall** | `{metrics.get('val_recall_macro', 0.0):.4f}` |",
            "",
        ])

    with open(summary_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def evaluate_and_register(
    model_name: str = DEFAULT_MODEL_NAME,
    override_threshold: Optional[float] = None,
    target_stage: str = "Staging",
    tracking_uri: Optional[str] = None,
) -> bool:
    """Execute evaluation gate and conditionally register model to Staging."""
    logger.info("=================================================================")
    logger.info("[EVALUATION GATE] EXECUTING AUTOMATED MODEL VALIDATION GATE")
    logger.info("=================================================================")

    # 1. Load Threshold & Metadata
    threshold = override_threshold if override_threshold is not None else load_threshold_from_config()
    metadata = load_latest_metadata()

    # Support both LK-06 flat schema and LK-07 nested schema
    run_id = metadata.get("best_run_id") or metadata.get("model_registry", {}).get("source_run_id")
    if not run_id:
        raise ValueError("Could not extract run ID from models/model_metadata.json")

    metrics = metadata.get("metrics") or metadata.get("evaluation_metrics", {})
    actual_f1 = float(metrics.get("val_f1_macro") or metrics.get("macro_f1", 0.0))
    classifier_type = (
        metadata.get("parameters", {}).get("classifier_type")
        or metadata.get("hyperparameters", {}).get("classifier_type", "linear_svc")
    )
    model_artifact = metadata.get("best_model_artifact_name") or f"{classifier_type}_classifier"
    resolved_tracking_uri = tracking_uri or metadata.get("tracking_uri", DEFAULT_TRACKING_URI)

    logger.info(f"Target Model Name   : '{model_name}'")
    logger.info(f"Source MLflow Run ID: '{run_id}'")
    logger.info(f"Performance Metric  : 'val_f1_macro' = {actual_f1:.4f}")
    logger.info(f"Required Threshold  : >= {threshold:.4f} (LK-01 Config)")
    logger.info(f"Tracking URI        : '{resolved_tracking_uri}'")
    logger.info("-" * 65)

    # 2. Check Threshold Gate
    passed = actual_f1 >= threshold

    if not passed:
        logger.error(
            f"[GATE REJECTED] Model validation FAILED! val_f1_macro ({actual_f1:.4f}) "
            f"is below threshold ({threshold:.4f})."
        )
        logger.error(
            "[SILENT FAILURE PREVENTED] Model will NOT be registered to Staging. "
            "Pipeline terminating with exit code 1."
        )
        write_github_step_summary(
            status="FAILED",
            metric_value=actual_f1,
            threshold=threshold,
            run_id=run_id,
            model_name=model_name,
            metrics=metrics,
        )
        return False

    logger.info(
        f"[GATE PASSED] Validation SUCCESS! val_f1_macro ({actual_f1:.4f}) >= threshold ({threshold:.4f})."
    )

    # 3. Auto-Registry & Promotion to Staging
    logger.info(f"\n[AUTO-REGISTRY] Registering model to MLflow Model Registry as '{model_name}'...")
    mlflow.set_tracking_uri(resolved_tracking_uri)
    client = MlflowClient(tracking_uri=resolved_tracking_uri)

    # Resolve model URI cleanly (supports MLflow 3.x logged models as well as classic runs:/ artifact URIs)
    model_uri = metadata.get("model_registry", {}).get("source_model_uri")
    if not model_uri:
        try:
            # Query MLflow logged models for this run
            exp = client.get_experiment_by_name(DEFAULT_EXPERIMENT_NAME)
            exp_ids = [exp.experiment_id] if exp else ["1"]
            logged_models = client.search_logged_models(experiment_ids=exp_ids)
            run_models = [lm for lm in logged_models if getattr(lm, "source_run_id", None) == run_id]
            if run_models:
                model_uri = run_models[0].model_uri
        except Exception as exc:
            logger.debug(f"Logged models lookup optional fallback: {exc}")

    if not model_uri:
        model_uri = f"runs:/{run_id}/{model_artifact}"

    logger.info(f"Resolved Model Source URI: '{model_uri}'")
    commit_sha = os.getenv("GITHUB_SHA", "local-test")
    actor = os.getenv("GITHUB_ACTOR", "automated-ci")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "ci_pipeline": "github-actions-mlops-automation",
            "evaluated_by": "evaluate_and_register.py",
            "gate_status": "PASSED",
            "threshold_f1": str(threshold),
            "actual_val_f1_macro": str(actual_f1),
            "git_commit": commit_sha,
            "triggered_by": actor,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    version_str = str(registered_model.version)
    logger.info(f"[TRANSITION] Promoting '{model_name}' v{version_str} -> Stage '{target_stage}'...")

    try:
        updated_mv = client.transition_model_version_stage(
            name=model_name,
            version=version_str,
            stage=target_stage,
            archive_existing_versions=False,
        )
        client.set_model_version_tag(
            name=model_name,
            version=version_str,
            key="automated_staging_at",
            value=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(
            f"[SUCCESS] Model '{model_name}' v{version_str} successfully promoted to stage '{updated_mv.current_stage}'!"
        )
    except Exception as exc:
        logger.warning(f"Note on stage transition: {exc}")

    write_github_step_summary(
        status="PASSED",
        metric_value=actual_f1,
        threshold=threshold,
        run_id=run_id,
        model_name=model_name,
        version=version_str,
        stage=target_stage,
        metrics=metrics,
    )

    logger.info("=================================================================")
    logger.info("[COMPLETE] TAHAP 3 & 4 EVALUATION & AUTO-REGISTRY FINISHED SUCCESSFULLY!")
    logger.info("=================================================================\n")
    return True


def main():
    parser = argparse.ArgumentParser(description="Automated Model Evaluation & Registry Gate")
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"Target Model Registry name (default: {DEFAULT_MODEL_NAME})",
    )
    parser.add_argument(
        "--override-threshold",
        type=float,
        default=None,
        help="Override performance threshold for negative testing (default: read from config.yaml)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Staging",
        help="Target stage for promoted model (default: Staging)",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI (default: read from metadata or sqlite:///mlruns.db)",
    )

    args = parser.parse_args()

    success = evaluate_and_register(
        model_name=args.model_name,
        override_threshold=args.override_threshold,
        target_stage=args.stage,
        tracking_uri=args.tracking_uri,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
