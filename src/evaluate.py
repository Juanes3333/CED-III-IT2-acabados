"""Evaluation metrics (accuracy, precision, recall, F1, Cohen's kappa) and model comparison.

Implements the per-split evaluation pipeline used by all notebooks:
  - compute_metrics   – 6-metric dict for one split (corrects slide 18 micro-average error)
  - confusion_matrix_data – raw 2×2 counts used by src/visualize.py (IT2-09)
  - evaluate_train_test – evaluates train AND test and returns a tidy DataFrame
  - save_metrics      – persists a metrics DataFrame to outputs/metrics/

bootstrap_ci and compare_models are implemented in IT2-24 (feat/lstm branch).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

SEED: int = 1234
ROOT: Path = Path(__file__).resolve().parents[1]
METRICS_DIR: Path = ROOT / "outputs" / "metrics"


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute classification metrics for binary (positive=1) sentiment labels.

    Corrects the slide 18 (p.45) error that uses ``average='micro'``: for
    single-label binary data micro precision/F1 collapse to accuracy and add
    no information.  We use ``average='binary'`` (positive class) for the
    primary metrics and also report ``average='macro'`` as required by the
    rubric (§3.3).

    Args:
        y_true: Ground-truth binary labels (1 = positive, 0 = negative).
            Accepts any array-like accepted by scikit-learn.
        y_pred: Predicted binary labels with the same shape as ``y_true``.

    Returns:
        Dictionary with six keys:
        ``accuracy``, ``precision``, ``recall``, ``f1`` (positive class,
        ``average='binary'``), ``f1_macro``, ``kappa`` (Cohen's kappa).
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def confusion_matrix_data(y_true, y_pred) -> dict[str, int]:
    """Return raw 2×2 confusion matrix values for a binary classifier.

    This helper exposes the counts that ``plot_confusion_matrix`` in
    ``src/visualize.py`` (IT2-09) needs without importing matplotlib here.

    Args:
        y_true: Ground-truth binary labels (0 = negative, 1 = positive).
        y_pred: Predicted binary labels with the same shape as ``y_true``.

    Returns:
        Dictionary with keys ``tn``, ``fp``, ``fn``, ``tp`` (all ``int``).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def evaluate_train_test(
    name: str,
    y_train,
    p_train,
    y_test,
    p_test,
) -> pd.DataFrame:
    """Evaluate a model on both the training and test splits.

    Calls ``compute_metrics`` for each split and assembles the results into a
    tidy DataFrame with one row per split.  Having both rows in one frame makes
    it easy to compute the train–test gap in IT2-24 (``compare_models``).

    Args:
        name: Human-readable model identifier used as the ``model`` column.
        y_train: True labels for the training split.
        p_train: Predicted labels for the training split.
        y_test: True labels for the test split.
        p_test: Predicted labels for the test split.

    Returns:
        DataFrame with columns ``model``, ``split``, ``accuracy``,
        ``precision``, ``recall``, ``f1``, ``f1_macro``, ``kappa``
        and two rows (``split`` = ``'train'`` and ``'test'``).
    """
    rows = []
    for split, y_true, y_pred in (
        ("train", y_train, p_train),
        ("test", y_test, p_test),
    ):
        metrics = compute_metrics(y_true, y_pred)
        rows.append({"model": name, "split": split, **metrics})
    return pd.DataFrame(rows)


def save_metrics(df: pd.DataFrame, name: str) -> Path:
    """Persist a metrics DataFrame to ``outputs/metrics/<name>_metrics.csv``.

    The output directory is created automatically.  Re-running overwrites the
    existing file, which is intentional: the final file always reflects the
    last experiment run.

    Args:
        df: Metrics DataFrame, typically from ``evaluate_train_test``.
        name: Model identifier used to build the filename
            (e.g. ``'baseline'`` → ``outputs/metrics/baseline_metrics.csv``).

    Returns:
        Absolute ``Path`` to the written CSV file.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS_DIR / f"{name}_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    return csv_path


# --- bootstrap_ci and compare_models are implemented in IT2-24 ---
