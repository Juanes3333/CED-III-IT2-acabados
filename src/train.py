"""Training utilities: reproducibility, logging, timing and grid-search results."""

from __future__ import annotations

import logging
import os
import random
import re
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

SEED = 1234


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for Python, NumPy and TensorFlow when available.

    TensorFlow deterministic operations are enabled when the installed version
    supports them. PYTHONHASHSEED is set for child processes; it cannot change
    the hash seed of the current Python process after startup.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required by set_seed(); activate the project environment."
        ) from exc

    tf.keras.utils.set_random_seed(seed)
    enable_determinism = getattr(tf.config.experimental, "enable_op_determinism", None)
    if callable(enable_determinism):
        enable_determinism()


def _safe_name(value: str) -> str:
    """Convert a model name to a filesystem-safe identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    if not cleaned:
        raise ValueError("The name must contain at least one alphanumeric character.")
    return cleaned


def get_logger(model_name: str, log_dir: str | Path = "logs") -> logging.Logger:
    """Return a model-specific logger writing to logs/training_logs_<model>.txt."""
    safe_model = _safe_name(model_name)
    log_path = Path(log_dir) / f"training_logs_{safe_model}.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"training.{safe_model}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Avoid duplicate handlers when this function is called repeatedly.
    handler_marker = str(log_path.resolve())
    if not any(getattr(handler, "_train_utils_path", None) == handler_marker
               for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )
        file_handler._train_utils_path = handler_marker
        logger.addHandler(file_handler)

    return logger


class Timer:
    """Context manager to measure elapsed time in seconds."""

    def __init__(self) -> None:
        self.elapsed_seconds = 0.0
        self._start: float | None = None

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds (alias for elapsed_seconds)."""
        return self.elapsed_seconds

    def __enter__(self) -> "Timer":
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._start is not None:
            self.elapsed_seconds = perf_counter() - self._start
            self._start = None


def save_grid_results(
    results: pd.DataFrame | list[dict[str, Any]],
    model_name: str,
    output_dir: str | Path = "outputs/metrics",
) -> Path:
    """Save hyperparameter-grid results as CSV and return the output path.

    Accepts either a Pandas DataFrame or a list of dictionaries.
    The file is named <model_name>_grid_results.csv.
    """
    safe_model = _safe_name(model_name)
    if isinstance(results, pd.DataFrame):
        dataframe = results.copy()
    else:
        dataframe = pd.DataFrame(results)

    if dataframe.empty:
        raise ValueError("Grid-search results are empty; nothing to save.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{safe_model}_grid_results.csv"
    dataframe.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path
