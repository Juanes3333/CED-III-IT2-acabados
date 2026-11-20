"""Training utilities: reproducibility, logging, timing and grid-search results."""

from __future__ import annotations

import logging
import os
import random
import re
from itertools import product
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

SEED = 1234


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for Python, NumPy and TensorFlow when available."""
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
    enable_determinism = getattr(
        tf.config.experimental, "enable_op_determinism", None
    )
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

    handler_marker = str(log_path.resolve())
    if not any(
        getattr(handler, "_train_utils_path", None) == handler_marker
        for handler in logger.handlers
    ):
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
    """Save hyperparameter-grid results as CSV and return the output path."""
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
    """Tune the TF-IDF + MLP pipeline with GridSearchCV on Cohen's kappa."""
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
        lambda params: __import__("json").dumps(
            params, default=_json_default, sort_keys=True
        )
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
        __import__("json").dumps(best, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "Dense grid search done | best_cv_kappa=%.4f | best_params=%s | seconds=%.1f",
        search.best_score_, search.best_params_, timer.elapsed,
    )
    logger.info("Saved %s and %s", csv_path, json_path)
    return search


# --- Keras sequence-model grid search ---

KERAS_PARAM_GRID: dict[str, list[Any]] = {
    "units": [32, 64, 128],
    "embedding_dim": [50, 100],
    "dropout": [0.2, 0.5],
    "batch_size": [16, 32, 64],
    "optimizer": ["RMSprop", "Adam"],
}


def _keras_model_params(params: dict[str, Any]) -> dict[str, Any]:
    """Translate shared grid names into model-builder parameter names."""
    translated = dict(params)
    if "units" in translated:
        translated["rnn_units"] = translated.pop("units")
    if "dropout" in translated:
        translated["dropout_rate"] = translated.pop("dropout")
    translated.pop("batch_size", None)
    translated.pop("optimizer", None)
    return translated


def _build_keras_candidate(
    build_fn,
    params: dict[str, Any],
    model_kwargs: dict[str, Any],
):
    """Build and compile one candidate using the selected optimizer."""
    import tensorflow as tf

    model_params = {**model_kwargs, **_keras_model_params(params)}
    model = build_fn(**model_params)

    optimizer_name = str(params.get("optimizer", "Adam")).lower()
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    elif optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer_name}. Use RMSprop or Adam."
        )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_keras_grid(
    build_fn,
    param_grid: dict,
    X_train,
    y_train,
    model_name: str,
    *,
    model_kwargs: dict[str, Any] | None = None,
    validation_size: float = 0.2,
    seed: int = SEED,
    output_dir: str | Path = "outputs/metrics",
    **fit_kwargs,
) -> pd.DataFrame:
    """Tune a Keras sequence model using a validation split from training data.

    Candidates are ranked by validation Cohen's kappa, with validation loss
    as the tie-breaker. The test set must never be passed to this function.
    """
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    import tensorflow as tf

    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")
    if not param_grid or any(len(values) == 0 for values in param_grid.values()):
        raise ValueError("param_grid must contain non-empty candidate values.")

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).reshape(-1)
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=seed,
        stratify=y_train,
    )

    grid_names = list(param_grid)
    configurations = [
        dict(zip(grid_names, values))
        for values in product(*(param_grid[name] for name in grid_names))
    ]

    epochs = int(fit_kwargs.pop("epochs", 30))
    verbose = fit_kwargs.pop("verbose", 0)
    logger = get_logger(model_name)
    logger.info(
        "Keras grid started | model=%s | candidates=%d | train=%d | "
        "validation=%d | seed=%d",
        model_name, len(configurations), len(X_fit), len(X_val), seed,
    )

    results: list[dict[str, Any]] = []

    for index, params in enumerate(configurations, start=1):
        set_seed(seed)
        model = _build_keras_candidate(
            build_fn=build_fn,
            params=params,
            model_kwargs=model_kwargs or {},
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            restore_best_weights=True,
        )

        logger.info("Candidate %d/%d | params=%s", index, len(configurations), params)
        with Timer() as timer:
            history = model.fit(
                X_fit,
                y_fit,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=int(params.get("batch_size", 32)),
                callbacks=[early_stopping],
                verbose=verbose,
                **fit_kwargs,
            )

        probabilities = model.predict(X_val, verbose=0).reshape(-1)
        predictions = (probabilities >= 0.5).astype(int)
        result = {
            **params,
            "val_loss": float(min(history.history["val_loss"])),
            "val_accuracy": float(accuracy_score(y_val, predictions)),
            "val_precision": float(
                precision_score(y_val, predictions, zero_division=0)
            ),
            "val_recall": float(recall_score(y_val, predictions, zero_division=0)),
            "val_f1": float(f1_score(y_val, predictions, zero_division=0)),
            "val_kappa": float(cohen_kappa_score(y_val, predictions)),
            "training_seconds": float(timer.elapsed),
            "trainable_parameters": int(model.count_params()),
            "epochs_trained": len(history.history["loss"]),
        }
        results.append(result)

        logger.info(
            "Candidate %d finished | kappa=%.4f | loss=%.4f | seconds=%.2f "
            "| parameters=%d",
            index,
            result["val_kappa"],
            result["val_loss"],
            timer.elapsed,
            result["trainable_parameters"],
        )
        del model
        tf.keras.backend.clear_session()

    results_df = pd.DataFrame(results).sort_values(
        ["val_kappa", "val_loss"],
        ascending=[False, True],
    ).reset_index(drop=True)

    csv_path = save_grid_results(results_df, model_name, output_dir)
    logger.info(
        "Keras grid completed | best_params=%s | best_val_kappa=%.4f",
        {key: results_df.iloc[0][key] for key in grid_names},
        results_df.iloc[0]["val_kappa"],
    )
    logger.info("Grid results saved to %s", csv_path)
    return results_df


def fit_best_keras(
    build_fn,
    best_params: dict,
    X_train,
    y_train,
    model_name: str,
    *,
    model_kwargs: dict[str, Any] | None = None,
    validation_split: float = 0.2,
    seed: int = SEED,
    **fit_kwargs,
):
    """Fit the selected Keras configuration using training data only.

    A stratified validation subset is held out from X_train for early stopping.
    Returns the fitted model and its Keras History object.
    """
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    if not 0 < validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1.")

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).reshape(-1)
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have the same number of rows.")

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=validation_split,
        random_state=seed,
        stratify=y_train,
    )

    set_seed(seed)
    model = _build_keras_candidate(
        build_fn=build_fn,
        params=best_params,
        model_kwargs=model_kwargs or {},
    )

    epochs = int(fit_kwargs.pop("epochs", 30))
    verbose = fit_kwargs.pop("verbose", 1)
    callbacks = list(fit_kwargs.pop("callbacks", []))
    callbacks.append(
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0.0001,
            restore_best_weights=True,
        )
    )

    logger = get_logger(model_name)
    logger.info("Refitting best Keras model | params=%s", best_params)

    with Timer() as timer:
        history = model.fit(
            X_fit,
            y_fit,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=int(best_params.get("batch_size", 32)),
            callbacks=callbacks,
            verbose=verbose,
            **fit_kwargs,
        )

    logger.info(
        "Best Keras model fitted | epochs=%d | seconds=%.2f | parameters=%d",
        len(history.history["loss"]),
        timer.elapsed,
        model.count_params(),
    )
    return model, history
