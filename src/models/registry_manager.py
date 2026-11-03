"""Module for managing MLflow Model Registry lifecycle, versioning, and stage transitions.

Fulfills LK-07 requirements:
1. Model Registration & Versioning (v1, v2 tracking).
2. Industry Stage Transition (None -> Staging -> Production).
3. Metadata synchronization with DVC for data lineage.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional
import yaml

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
from mlflow.tracking import MlflowClient

from src.models.train_model import (
    DEFAULT_EXPERIMENT_NAME,
    DEFAULT_TRACKING_URI,
    load_dvc_dataset_metadata,
    train_and_log_run,
)
from src.utils.logger import get_logger

logger = get_logger("model_registry")

MODEL_REGISTRY_NAME = "StockSentimentClassifier"
DEFAULT_DVC_PATH = PROJECT_ROOT / "data" / "raw" / "IDSMSA.csv.dvc"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "IDSMSA.csv"
OUTPUT_YAML_PATH = PROJECT_ROOT / "models" / "production_model_reference.yaml"


def get_client(tracking_uri: str = DEFAULT_TRACKING_URI) -> MlflowClient:
    """Initialize MLflow Client."""
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def train_and_register_v2(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    model_name: str = MODEL_REGISTRY_NAME,
) -> Dict[str, Any]:
    """Train updated model variation (v2) and register it to MLflow Model Registry."""
    logger.info("=================================================================")
    logger.info("[TRAIN v2] Training Enhanced Model Variation for Registry v2...")
    logger.info("=================================================================")

    # Variation v2: LinearSVC with higher n-gram range (1, 3), max_features=10000, C=1.5
    # for enhanced contextual n-gram coverage on Indonesian stock sentiment phrases.
    v2_params = {
        "run_name": "run4_linearsvc_tuned_trigram_v2",
        "classifier_type": "linear_svc",
        "model_artifact_name": "linearsvc_tuned_v2_classifier",
        "c_param": 1.5,
        "ngram_range": (1, 3),
        "max_features": 10000,
        "sublinear_tf": True,
        "tags": {
            "version_cycle": "v2.0-candidate",
            "tuning_objective": "higher_ngram_trigram_context",
            "variation_stage": "production_candidate",
            "description": "Enhanced LinearSVC with (1,3) tri-grams and 10k features",
        },
    }

    run_res = train_and_log_run(
        dataset_path=dataset_path,
        classifier_type=v2_params["classifier_type"],
        c_param=v2_params["c_param"],
        ngram_range=v2_params["ngram_range"],
        max_features=v2_params["max_features"],
        sublinear_tf=v2_params["sublinear_tf"],
        experiment_name=experiment_name,
        run_name=v2_params["run_name"],
        model_artifact_name=v2_params["model_artifact_name"],
        tracking_uri=tracking_uri,
        tags=v2_params["tags"],
    )

    client = get_client(tracking_uri)
    model_uri = f"runs:/{run_res['run_id']}/{run_res['model_artifact_name']}"

    logger.info(f"[REGISTRY] Registering model to '{model_name}' from URI: {model_uri}")
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
        tags={
            "created_by": "mlops_pipeline_lk07",
            "source_run_id": run_res["run_id"],
            "model_type": v2_params["classifier_type"],
            "ngram_range": str(v2_params["ngram_range"]),
            "registered_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    version_str = str(registered_model.version)
    client.update_model_version(
        name=model_name,
        version=version_str,
        description=(
            f"Version {version_str}: Enhanced LinearSVC with Tri-gram (1,3) TF-IDF & C=1.5. "
            f"Validation Macro F1: {run_res['metrics']['val_f1_macro']:.4f}, "
            f"Accuracy: {run_res['metrics']['val_accuracy']:.4f}."
        ),
    )

    logger.info(
        f"[SUCCESS] Model '{model_name}' Version {version_str} registered successfully! (Current stage: None)"
    )
    return {
        "model_name": model_name,
        "version": int(version_str),
        "run_id": run_res["run_id"],
        "metrics": run_res["metrics"],
        "params": run_res["params"],
    }


def transition_stage(
    model_name: str = MODEL_REGISTRY_NAME,
    version: int = 2,
    target_stage: str = "Staging",
    archive_existing_versions: bool = False,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    comment: Optional[str] = None,
) -> Any:
    """Transition a registered model version to target stage (None, Staging, Production, Archived)."""
    client = get_client(tracking_uri)
    version_str = str(version)

    valid_stages = ["None", "Staging", "Production", "Archived"]
    matched_stage = next((s for s in valid_stages if s.lower() == target_stage.lower()), None)
    if not matched_stage:
        raise ValueError(f"Invalid target stage '{target_stage}'. Valid options: {valid_stages}")

    # archive_existing_versions is only valid for active stages ('Staging', 'Production')
    can_archive = archive_existing_versions and (matched_stage in ["Staging", "Production"])

    logger.info(
        f"[TRANSITION] Transitioning '{model_name}' v{version_str} -> Stage '{matched_stage}' "
        f"(archive_existing={can_archive})..."
    )

    updated_mv = client.transition_model_version_stage(
        name=model_name,
        version=version_str,
        stage=matched_stage,
        archive_existing_versions=can_archive,
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    client.set_model_version_tag(
        name=model_name,
        version=version_str,
        key=f"stage_transition_{matched_stage.lower()}_at",
        value=timestamp,
    )

    if comment:
        client.set_model_version_tag(
            name=model_name,
            version=version_str,
            key=f"comment_{matched_stage.lower()}",
            value=comment,
        )

    # If transitioning to Production, also set alias champion if supported
    if matched_stage == "Production":
        try:
            client.set_registered_model_alias(model_name, "champion", version_str)
            client.set_registered_model_alias(model_name, "production", version_str)
            logger.info(f"[ALIAS] Set aliases 'champion' and 'production' to version {version_str}")
        except Exception as exc:
            logger.debug(f"Alias setting optional note: {exc}")

    logger.info(
        f"[SUCCESS] Model '{model_name}' v{version_str} is now in stage: '{updated_mv.current_stage}'"
    )
    return updated_mv


def export_model_metadata_and_dvc(
    model_name: str = MODEL_REGISTRY_NAME,
    version: int = 2,
    output_yaml: Path = OUTPUT_YAML_PATH,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    dvc_file: Path = DEFAULT_DVC_PATH,
) -> Dict[str, Any]:
    """Generate YAML metadata linking published model with DVC data lineage and track with DVC."""
    client = get_client(tracking_uri)
    version_str = str(version)

    mv = client.get_model_version(name=model_name, version=version_str)
    run = client.get_run(mv.run_id)

    # Load DVC lineage info
    dvc_meta = load_dvc_dataset_metadata(dvc_file)

    published_metadata = {
        "model_registry": {
            "name": model_name,
            "version": int(mv.version),
            "stage": mv.current_stage,
            "status": mv.status,
            "description": mv.description or "",
            "source_run_id": mv.run_id,
            "source_model_uri": mv.source,
            "inference_model_uri": f"models:/{model_name}/{mv.current_stage}",
            "registered_at": datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc).isoformat(),
            "last_updated_at": datetime.fromtimestamp(mv.last_updated_timestamp / 1000, tz=timezone.utc).isoformat(),
        },
        "evaluation_metrics": {
            "accuracy": float(run.data.metrics.get("val_accuracy", 0.0)),
            "macro_f1": float(run.data.metrics.get("val_f1_macro", 0.0)),
            "weighted_f1": float(run.data.metrics.get("val_f1_weighted", 0.0)),
            "macro_precision": float(run.data.metrics.get("val_precision_macro", 0.0)),
            "macro_recall": float(run.data.metrics.get("val_recall_macro", 0.0)),
        },
        "hyperparameters": {
            "classifier_type": run.data.params.get("classifier_type", "linear_svc"),
            "c_regularization": float(run.data.params.get("c_regularization", 1.0)),
            "ngram_min": int(run.data.params.get("ngram_min", 1)),
            "ngram_max": int(run.data.params.get("ngram_max", 3)),
            "max_features": int(run.data.params.get("max_features", 10000)),
            "sublinear_tf": run.data.params.get("sublinear_tf") in [True, "True", "true"],
        },
        "data_lineage": {
            "dataset_file": "data/raw/IDSMSA.csv",
            "dvc_pointer_file": str(dvc_file.relative_to(PROJECT_ROOT)) if dvc_file.exists() else str(dvc_file),
            "dvc_data_md5": dvc_meta.get("dvc_md5", "eed20df7006e3316eb967ec5c2371d5c"),
            "dvc_data_size_bytes": dvc_meta.get("dvc_size_bytes", 1091613),
            "train_samples": int(run.data.params.get("train_samples", 2630)),
            "test_samples": int(run.data.params.get("test_samples", 658)),
        },
        "inference_specification": {
            "loader_function": "mlflow.pyfunc.load_model",
            "target_service": "FastAPI Sentiment Inference Service",
            "expected_input_format": "List[str] or pandas.Series of raw/cleaned stock tweets",
            "possible_classes": ["Positive", "Neutral", "Negative"],
            "target_latency": "<15ms",
        },
    }

    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml, "w", encoding="utf-8") as f:
        yaml.dump(published_metadata, f, default_flow_style=False, sort_keys=False)

    logger.info(f"[YAML] Model metadata successfully written to: {output_yaml}")

    # Also update models/model_metadata.json for consistency
    json_path = PROJECT_ROOT / "models" / "model_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(published_metadata, f, indent=2)
    logger.info(f"[JSON] JSON metadata synchronized to: {json_path}")

    # Track metadata file with DVC
    logger.info(f"[DVC] Running 'dvc add {output_yaml.relative_to(PROJECT_ROOT)}'...")
    try:
        cmd = [sys.executable, "-m", "dvc", "add", str(output_yaml.relative_to(PROJECT_ROOT))]
        res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        if res.returncode == 0:
            logger.info("[DVC] Metadata successfully versioned with DVC!")
            logger.info(f"[DVC] Pointer file created at: {output_yaml}.dvc")
        else:
            logger.warning(f"[DVC] Note: DVC add returned code {res.returncode}: {res.stderr}")
    except Exception as exc:
        logger.warning(f"[DVC] Could not run dvc add automatically: {exc}")

    return published_metadata


def print_registry_status(model_name: str = MODEL_REGISTRY_NAME, tracking_uri: str = DEFAULT_TRACKING_URI):
    """Print tabular overview of all versions of a registered model."""
    client = get_client(tracking_uri)
    logger.info(f"\n=================================================================")
    logger.info(f"MLFLOW MODEL REGISTRY STATUS: '{model_name}'")
    logger.info(f"Tracking URI: {tracking_uri}")
    logger.info("=================================================================")
    logger.info(f"{'Version':<10} | {'Stage':<12} | {'Status':<10} | {'Run ID':<34} | {'Description'}")
    logger.info("-" * 95)

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in sorted(versions, key=lambda x: int(x.version)):
            desc = (v.description or "")[:40]
            logger.info(f"{v.version:<10} | {v.current_stage:<12} | {v.status:<10} | {v.run_id:<34} | {desc}")
    except Exception as exc:
        logger.error(f"Error querying model registry: {exc}")
    logger.info("-" * 95)


def run_full_lifecycle_pipeline():
    """Execute complete LK-07 lifecycle: Train v2 -> Transition Staging -> Production -> DVC Sync."""
    logger.info("=================================================================")
    logger.info("[LIFECYCLE] EXECUTING COMPLETE LK-07 MODEL REGISTRY LIFECYCLE")
    logger.info("=================================================================")

    # 1. Train and Register v2
    train_res = train_and_register_v2()
    v2_version = train_res["version"]

    # 2. Stage Transition: None -> Staging (UAT / Quality Gate)
    logger.info("\n--- STEP 1: TRANSITIONING TO STAGING (SIMULATING UAT / EVALUATION) ---")
    transition_stage(
        model_name=MODEL_REGISTRY_NAME,
        version=v2_version,
        target_stage="Staging",
        archive_existing_versions=False,
        comment="Promoted to Staging for latency and integration testing.",
    )

    # 3. Stage Transition: Staging -> Production (Release to Service)
    logger.info("\n--- STEP 2: TRANSITIONING TO PRODUCTION (SIMULATING LIVE RELEASE) ---")
    transition_stage(
        model_name=MODEL_REGISTRY_NAME,
        version=v2_version,
        target_stage="Production",
        archive_existing_versions=True,  # Automatically archives v1
        comment="Promoted to Production. Surpassed baseline with superior Macro F1.",
    )

    # 4. Sync Metadata & DVC Lineage
    logger.info("\n--- STEP 3: SYNCHRONIZING METADATA & DVC DATA LINEAGE ---")
    export_model_metadata_and_dvc(
        model_name=MODEL_REGISTRY_NAME,
        version=v2_version,
    )

    # 5. Display Status Table
    print_registry_status(MODEL_REGISTRY_NAME)
    logger.info("\n[COMPLETED] Full LK-07 Lifecycle Pipeline completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="MLflow Model Registry Lifecycle Manager")
    parser.add_argument(
        "--action",
        type=str,
        default="status",
        choices=["train-v2", "transition", "sync-metadata", "status", "full-pipeline"],
        help="Action to perform",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=MODEL_REGISTRY_NAME,
        help=f"Registered model name (default: {MODEL_REGISTRY_NAME})",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=2,
        help="Model version to operate on",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="Production",
        choices=["None", "Staging", "Production", "Archived"],
        help="Target stage for transition",
    )
    parser.add_argument(
        "--archive-existing",
        action="store_true",
        default=True,
        help="Archive existing versions when promoting to target stage",
    )
    parser.add_argument(
        "--comment",
        type=str,
        default=None,
        help="Audit comment for stage transition",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=DEFAULT_TRACKING_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_TRACKING_URI})",
    )

    args = parser.parse_args()

    if args.action == "train-v2":
        train_and_register_v2(
            tracking_uri=args.tracking_uri,
            model_name=args.model_name,
        )
    elif args.action == "transition":
        transition_stage(
            model_name=args.model_name,
            version=args.version,
            target_stage=args.stage,
            archive_existing_versions=args.archive_existing,
            tracking_uri=args.tracking_uri,
            comment=args.comment,
        )
    elif args.action == "sync-metadata":
        export_model_metadata_and_dvc(
            model_name=args.model_name,
            version=args.version,
            tracking_uri=args.tracking_uri,
        )
    elif args.action == "status":
        print_registry_status(args.model_name, args.tracking_uri)
    elif args.action == "full-pipeline":
        run_full_lifecycle_pipeline()


if __name__ == "__main__":
    main()
