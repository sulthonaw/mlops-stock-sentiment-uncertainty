"""Dynamic Text Preprocessing & Cleaning Automation for Stock Tweets.

Standardizes raw stock tweets to ID-SMSA baseline specifications:
- Replaces user mentions with [USERNAME]
- Replaces URLs with [URL]
- Replaces hashtags with [HASHTAG]
- Cleans duplicate whitespace and newline characters
- Drops null and empty sentences
- Removes duplicate sentences
- Exports to data/interim in Parquet and CSV formats with batch traceability.
"""

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data_preprocessing")

# Regex patterns consistent with ID-SMSA baseline standard
URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_REGEX = re.compile(r"@[A-Za-z0-9_]+", re.IGNORECASE)
HASHTAG_REGEX = re.compile(r"#[A-Za-z0-9_]+", re.IGNORECASE)
WHITESPACE_REGEX = re.compile(r"\s+")


def clean_text_idsmsa(text: Any) -> str:
    """Clean a single text string according to ID-SMSA baseline standards."""
    if text is None or pd.isna(text):
        return ""

    text = str(text)

    # 1. URL normalization
    text = URL_REGEX.sub("[URL]", text)

    # 2. Account mention normalization
    text = MENTION_REGEX.sub("[USERNAME]", text)

    # 3. Hashtag normalization
    text = HASHTAG_REGEX.sub("[HASHTAG]", text)

    # 4. Collapse consecutive whitespace and newlines, then strip
    text = WHITESPACE_REGEX.sub(" ", text).strip()

    return text


def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Clean, filter, and deduplicate a dataframe of tweets."""
    initial_count = len(df)

    # Validate required text column
    if "Sentence" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'Sentence' column for text preprocessing."
        )

    df_cleaned = df.copy()

    # Drop strictly null sentence rows before cleaning
    null_count_before = int(df_cleaned["Sentence"].isna().sum())
    df_cleaned = df_cleaned.dropna(subset=["Sentence"]).reset_index(drop=True)

    # Apply text normalization
    df_cleaned["Sentence"] = df_cleaned["Sentence"].apply(clean_text_idsmsa)

    # Drop empty or whitespace-only strings
    empty_mask = df_cleaned["Sentence"].str.len() == 0
    empty_count = int(empty_mask.sum())
    df_cleaned = df_cleaned[~empty_mask].reset_index(drop=True)

    missing_total = null_count_before + empty_count

    # Deduplicate based on cleaned Sentence
    count_before_dedup = len(df_cleaned)
    df_cleaned = df_cleaned.drop_duplicates(
        subset=["Sentence"], keep="first"
    ).reset_index(drop=True)
    duplicates_removed = count_before_dedup - len(df_cleaned)

    final_count = len(df_cleaned)

    stats = {
        "initial_rows": initial_count,
        "missing_removed": missing_total,
        "duplicates_removed": duplicates_removed,
        "cleaned_rows": final_count,
    }

    return df_cleaned, stats


def find_latest_raw_batch(input_dir: Path) -> Optional[Path]:
    """Find the most recent raw stream batch file or raw crawl CSV."""
    # Look for raw_stream_*.json first (excluding companion metadata files)
    json_streams = [
        f
        for f in sorted(glob.glob(str(input_dir / "raw_stream_*.json")), reverse=True)
        if not f.endswith("_metadata.json")
    ]
    if json_streams:
        return Path(json_streams[0])

    # Fallback to raw_stream_*.csv
    csv_streams = sorted(glob.glob(str(input_dir / "raw_stream_*.csv")), reverse=True)
    if csv_streams:
        return Path(csv_streams[0])

    # Fallback to crawl CSVs
    crawl_csvs = sorted(
        glob.glob(str(input_dir / "x_stock_tweets_*.csv")), reverse=True
    )
    if crawl_csvs:
        return Path(crawl_csvs[0])

    return None


def load_input_data(input_path: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Load dataframe and associated metadata from JSON or CSV raw file."""
    metadata: Dict[str, Any] = {}
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading raw batch from: {input_path}")

    if input_path.suffix.lower() == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "records" in data:
            metadata = data.get("metadata", {})
            df = pd.DataFrame(data["records"])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
    elif input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
        meta_companion = input_path.with_name(f"{input_path.stem}_metadata.json")
        if meta_companion.exists():
            with open(meta_companion, "r", encoding="utf-8") as f:
                metadata = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

    return df, metadata


def preprocess_data(
    input_path: Optional[Path] = None,
    input_dir: Path = Path("data/raw"),
    output_dir: Path = Path("data/interim"),
    output_format: str = "both",
) -> Dict[str, Any]:
    """Execute text preprocessing pipeline and export cleaned dataset."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path is None:
        latest = find_latest_raw_batch(input_dir)
        if latest is None:
            raise FileNotFoundError(f"No raw files found in: {input_dir}")
        input_path = latest

    input_path = Path(input_path)
    df_raw, raw_metadata = load_input_data(input_path)

    logger.info(f"Starting preprocessing on {len(df_raw)} raw rows...")
    df_clean, stats = preprocess_dataframe(df_raw)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    source_stem = input_path.stem.replace("raw_stream_", "").replace(
        "x_stock_tweets_", ""
    )
    base_name = f"interim_stream_{source_stem}_{timestamp_str}"

    processed_metadata = {
        "processed_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_batch_file": input_path.name,
        "source_metadata": raw_metadata,
        "statistics": stats,
        "output_columns": list(df_clean.columns),
    }

    exported_files = []

    if output_format in ["parquet", "both"]:
        parquet_file = output_dir / f"{base_name}.parquet"
        df_clean.to_parquet(parquet_file, index=False, engine="pyarrow")
        exported_files.append(str(parquet_file))
        logger.info(f"Exported Parquet interim data to: {parquet_file}")

    if output_format in ["csv", "both"]:
        csv_file = output_dir / f"{base_name}.csv"
        df_clean.to_csv(csv_file, index=False, encoding="utf-8")
        exported_files.append(str(csv_file))
        logger.info(f"Exported CSV interim data to: {csv_file}")

    # Export preprocessing metadata
    meta_file = output_dir / f"{base_name}_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(processed_metadata, f, ensure_ascii=False, indent=2)
    exported_files.append(str(meta_file))

    # Print summary log
    print("\n" + "=" * 60)
    print("           RINGKASAN PRAPEMROSESAN DATA (LK-04)")
    print("=" * 60)
    print(f"Berkas Sumber        : {input_path.name}")
    print(f"Jumlah Baris Awal    : {stats['initial_rows']}")
    print(f"Missing Values Dibuang: {stats['missing_removed']}")
    print(f"Duplikat Dihapus     : {stats['duplicates_removed']}")
    print(f"Total Baris Bersih   : {stats['cleaned_rows']}")
    print(f"Format Ekspor        : {output_format}")
    print(f"Direktori Output     : {output_dir.resolve()}")
    print("=" * 60 + "\n")

    return {
        "status": "success",
        "statistics": stats,
        "exported_files": exported_files,
    }


def main():
    """CLI entrypoint for data preprocessing."""
    parser = argparse.ArgumentParser(
        description="Automate text cleaning and standardization to ID-SMSA baseline specifications."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to specific raw stream file (e.g. data/raw/raw_stream_20260928_214500.json)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Directory to search for raw files if --input-path is omitted (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/interim",
        help="Directory to save cleaned interim files (default: data/interim)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv", "both"],
        default="both",
        help="Output serialization format (default: both)",
    )

    args = parser.parse_args()

    preprocess_data(
        input_path=Path(args.input_path) if args.input_path else None,
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        output_format=args.format,
    )


if __name__ == "__main__":
    main()
