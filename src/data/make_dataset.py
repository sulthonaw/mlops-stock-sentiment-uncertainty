"""Dataset creation and preprocessing module."""

from pathlib import Path
from typing import Optional
from src.preprocess import clean_text_idsmsa, preprocess_data, preprocess_dataframe

__all__ = ["clean_text_idsmsa", "preprocess_dataframe", "preprocess_data", "main"]


def main(
    input_path: Optional[str] = None,
    output_dir: str = "data/interim",
    format: str = "both",
):
    """Entrypoint to run dataset preprocessing pipeline."""
    preprocess_data(
        input_path=Path(input_path) if input_path else None,
        output_dir=Path(output_dir),
        output_format=format,
    )


if __name__ == "__main__":
    main()
