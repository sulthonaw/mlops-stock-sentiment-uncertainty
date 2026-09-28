"""Dynamic Data Ingestion Script for Stock Sentiment Analysis.

Scans, validates, and consolidates raw crawl tweets into non-destructive
timestamped raw stream batches with rich metadata and optional mock data generation.
"""

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("data_ingestion")

EXPECTED_COLUMNS = [
    "Sentence",
    "Sentiment",
    "Tweet Date",
    "English Translation",
    "Favorite Count",
    "Retweet Count",
    "Reply Count",
    "Quote Count",
]

TARGET_TICKERS = [
    "BBCA",
    "BBRI",
    "BMRI",
    "BBNI",
    "TLKM",
    "ASII",
    "UNVR",
    "TPIA",
    "HMSP",
    "BYAN",
]


def detect_tickers(sentences: pd.Series) -> List[str]:
    """Detect mentioned stock tickers from tweet sentences."""
    detected = set()
    ticker_pattern = re.compile(
        r"(?:\$|\b)(" + "|".join(TARGET_TICKERS) + r")\b", re.IGNORECASE
    )
    for text in sentences.dropna():
        matches = ticker_pattern.findall(str(text))
        for m in matches:
            detected.add(m.upper())
    return sorted(list(detected))


def generate_mock_tweets(count: int = 10) -> pd.DataFrame:
    """Generate realistic synthetic tweets for testing ingestion dynamics."""
    now_str = datetime.now(timezone.utc).strftime("%a %b %d %H:%M:%S +0000 %Y")
    mock_samples = [
        (
            "Kinerja $BBCA kuartal ini luar biasa solid, dividen interim siap ditunggu investor.",
            "BBCA",
        ),
        (
            "Sinyal akumulasi asing terlihat masif di $BBRI setelah breakout resistance 3200.",
            "BBRI",
        ),
        ("Laporan laba bersih $BMRI melesat melampaui konsensus pasar modal.", "BMRI"),
        (
            "Valuasi $BBNI masih relatif undervalue dibanding perbankan big four lainnya.",
            "BBNI",
        ),
        (
            "Transformasi bisnis data center $TLKM mulai membuahkan hasil positif jangka panjang.",
            "TLKM",
        ),
        (
            "Penjualan kendaraan listrik diprediksi dorong rebound margin laba $ASII.",
            "ASII",
        ),
        (
            "Restrukturisasi distribusi $UNVR mulai menunjukkan tanda-tanda pemulihan konsumsi ritel.",
            "UNVR",
        ),
        (
            "Ekspansi pabrik petrokimia $TPIA berpeluang perkuat kapasitas produksi domestik.",
            "TPIA",
        ),
        (
            "Volume perdagangan $HMSP hari ini meningkat tajam dengan net buy domestik.",
            "HMSP",
        ),
        (
            "Lonjakan permintaan batubara global memberikan sentimen positif bagi pergerakan $BYAN.",
            "BYAN",
        ),
    ]

    records = []
    for i in range(count):
        sample_text, _ = mock_samples[i % len(mock_samples)]
        records.append(
            {
                "Sentence": f"[SIMULASI MOCK] {sample_text} (Batch stream id #{i+1})",
                "Sentiment": None,
                "Tweet Date": now_str,
                "English Translation": None,
                "Favorite Count": (i + 1) * 3,
                "Retweet Count": (i + 1) * 2,
                "Reply Count": i,
                "Quote Count": 0,
            }
        )
    return pd.DataFrame(records)


def scan_and_validate_files(input_dir: Path) -> Tuple[List[Path], List[pd.DataFrame]]:
    """Scan directory for crawled CSV files and validate their schema."""
    crawl_patterns = [
        input_dir / "x_stock_tweets_*.csv",
    ]
    matched_files: List[Path] = []
    for pat in crawl_patterns:
        for f in glob.glob(str(pat)):
            p = Path(f)
            if p.is_file():
                matched_files.append(p)

    matched_files = sorted(list(set(matched_files)))
    valid_dfs: List[pd.DataFrame] = []
    valid_files: List[Path] = []

    logger.info(f"Scanning directory: {input_dir.resolve()}")
    logger.info(f"Found {len(matched_files)} candidate raw crawl file(s).")

    for file_path in matched_files:
        try:
            df = pd.read_csv(file_path)
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            if missing_cols:
                logger.warning(
                    f"File '{file_path.name}' is missing expected columns: {missing_cols}. Skipping."
                )
                continue

            # Ensure expected column order and presence
            df = df[EXPECTED_COLUMNS].copy()
            valid_dfs.append(df)
            valid_files.append(file_path)
            logger.info(
                f"Successfully validated '{file_path.name}' ({len(df)} records)."
            )
        except Exception as e:
            logger.error(f"Failed to read/validate '{file_path.name}': {e}")

    return valid_files, valid_dfs


def ingest_data(
    input_dir: Path,
    output_dir: Path,
    output_format: str = "both",
    mock_new: bool = False,
    mock_count: int = 10,
) -> Dict[str, Any]:
    """Execute dynamic ingestion pipeline and save timestamped raw stream."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_files, dfs = scan_and_validate_files(input_dir)

    all_dfs = list(dfs)
    mock_added = 0
    if mock_new:
        logger.info(f"Generating {mock_count} mock new stream records...")
        mock_df = generate_mock_tweets(count=mock_count)
        all_dfs.append(mock_df)
        mock_added = len(mock_df)

    if not all_dfs:
        logger.warning("No data found or generated to ingest.")
        return {"status": "no_data"}

    consolidated_df = pd.concat(all_dfs, ignore_index=True)
    detected_tickers = detect_tickers(consolidated_df["Sentence"])

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    iso_timestamp = datetime.now(timezone.utc).isoformat()

    metadata = {
        "ingestion_timestamp": iso_timestamp,
        "batch_id": f"batch_{timestamp_str}",
        "total_records": len(consolidated_df),
        "mock_records_added": mock_added,
        "tickers": detected_tickers,
        "schema_version": "1.0.0",
        "source_files": [f.name for f in valid_files],
        "columns": EXPECTED_COLUMNS,
    }

    base_name = f"raw_stream_{timestamp_str}"
    exported_files = []

    if output_format in ["json", "both"]:
        json_file = output_dir / f"{base_name}.json"
        payload = {
            "metadata": metadata,
            "records": consolidated_df.to_dict(orient="records"),
        }
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        exported_files.append(str(json_file))
        logger.info(f"Exported JSON stream batch to: {json_file}")

    if output_format in ["csv", "both"]:
        csv_file = output_dir / f"{base_name}.csv"
        meta_file = output_dir / f"{base_name}_metadata.json"
        consolidated_df.to_csv(csv_file, index=False, encoding="utf-8")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        exported_files.append(str(csv_file))
        exported_files.append(str(meta_file))
        logger.info(f"Exported CSV stream batch to: {csv_file}")

    logger.info(
        f"Ingestion summary: {len(consolidated_df)} records consolidated "
        f"across {len(valid_files)} source files and {mock_added} mock records. "
        f"Tickers: {', '.join(detected_tickers)}"
    )

    return {
        "status": "success",
        "metadata": metadata,
        "exported_files": exported_files,
        "record_count": len(consolidated_df),
    }


def main():
    """CLI entrypoint for data ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest dynamic crawled tweets and export non-destructive timestamped stream batches."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw crawl CSV files (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save timestamped raw stream batches (default: data/raw)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "both"],
        default="both",
        help="Output serialization format (default: both)",
    )
    parser.add_argument(
        "--mock-new",
        action="store_true",
        help="Simulate arrival of new real-time tweets for testing periodic ingestion",
    )
    parser.add_argument(
        "--mock-count",
        type=int,
        default=10,
        help="Number of mock records to synthesize if --mock-new is set (default: 10)",
    )

    args = parser.parse_args()

    ingest_data(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        output_format=args.format,
        mock_new=args.mock_new,
        mock_count=args.mock_count,
    )


if __name__ == "__main__":
    main()
