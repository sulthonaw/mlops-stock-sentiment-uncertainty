"""Standard logging utility for MLOps Stock Sentiment project."""

import logging
import sys


def get_logger(name: str = "mlops") -> logging.Logger:
    """Configure and return a standardized logger instance."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Handle UTF-8 encoding safely across platforms (including Windows cp1252)
        if sys.platform == "win32":
            try:
                import io
                stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
            except Exception:
                stream = sys.stdout
        else:
            stream = sys.stdout

        handler = logging.StreamHandler(stream)
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
