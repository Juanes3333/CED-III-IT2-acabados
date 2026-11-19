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


# --- Dense (sklearn) grid search ---

# Grid from MASTER_PROMPT.md section 6. Keys use the Pipeline "step__param" names
# of build_dense_pipeline(), so they can be passed to GridSearchCV as they are.
DENSE_PARAM_GRID: dict[str, list[Any]] = {
    "mlp__hidden_layer_sizes": [(32,), (64,), (128,), (64, 32)],
    "mlp__activation": ["relu", "tanh"],
    "mlp__alpha": [1e-4, 1e-3, 1e-2],
    "mlp__learning_rate_init": [1e-3, 1e-2],
}


def _json_default(value: Any) -> Any:
    """Make NumPy values and tuples readable by json.dump."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (tuple, set)):
        return list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def run_dense_grid(
    X_train,
    y_train,
    param_grid: dict,
    cv: int = 5,
    *,
    output_dir: str | Path = "outputs/metrics",
    n_jobs: int | None = None,
):
    """Tune the TF-IDF + MLP pipeline with GridSearchCV on Cohen's kappa.

    Only the training texts and labels are passed in, so the test set is never
    used for tuning. The vectorizer is inside the pipeline: each CV fold learns
    its own TF-IDF vocabulary from the fold's training rows only.

    Args:
        X_train: Training texts (for example ``train["clean_text"]``).
        y_train: Training labels.
        param_grid: ``GridSearchCV`` grid with ``step__param`` keys, such as
            ``mlp__hidden_layer_sizes``. See ``DENSE_PARAM_GRID``.
        cv: Number of stratified folds.
        output_dir: Folder for ``dense_grid_results.csv`` and
            ``dense_best_params.json``.
        n_jobs: Parallel jobs for ``GridSearchCV`` (``None`` = one job).

    Returns:
        The fitted ``GridSearchCV`` (``refit=True``, so ``best_estimator_`` has
        been refitted on the whole training split).
    """
    import json

    from sklearn.metrics import cohen_kappa_score, make_scorer
    from sklearn.model_selection import GridSearchCV, StratifiedKFold

    from src.models import build_dense_pipeline

    logger = get_logger("dense")
    n_candidates = int(np.prod([len(values) for values in param_grid.values()]))
    logger.info(
        "Dense grid search start | rows=%d | candidates=%d | cv=%d | seed=%d",
        len(X_train), n_candidates, cv, SEED,
    )

    # make_scorer turns the metric into a scorer; passing the bare function
    # (slide 18) would fail because scorers receive (estimator, X, y).
    kappa_scorer = make_scorer(cohen_kappa_score)
    folds = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)
    search = GridSearchCV(
        estimator=build_dense_pipeline(),
        param_grid=param_grid,
        scoring=kappa_scorer,
        cv=folds,
        refit=True,
        n_jobs=n_jobs,
        return_train_score=True,
    )

    with Timer() as timer:
        search.fit(X_train, y_train)

    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    results["params"] = results["params"].apply(
        lambda params: json.dumps(params, default=_json_default, sort_keys=True)
    )
    csv_path = save_grid_results(results, "dense", output_dir)

    json_path = Path(output_dir) / "dense_best_params.json"
    best = {
        "best_params": search.best_params_,
        "best_cv_kappa": float(search.best_score_),
        "scoring": "cohen_kappa_score",
        "cv_folds": int(cv),
        "n_candidates": n_candidates,
        "n_train_rows": int(len(X_train)),
        "seed": SEED,
    }
    json_path.write_text(
        json.dumps(best, indent=2, default=_json_default) + "\n", encoding="utf-8"
    )

    logger.info(
        "Dense grid search done | best_cv_kappa=%.4f | best_params=%s | seconds=%.1f",
        search.best_score_, search.best_params_, timer.elapsed,
    )
    logger.info("Saved %s and %s", csv_path, json_path)
    return search
