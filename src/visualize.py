"""Plotting helpers for EDA and model evaluation results."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "outputs" / "figures"

sns.set_theme(style="whitegrid", palette="deep")


def _save(fig: Figure, name: str) -> Figure:
    """Save a figure as a 150-dpi PNG and return it."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=150, bbox_inches="tight")
    return fig


def _group_column(df: pd.DataFrame, by: str) -> str:
    """Use the requested group column, falling back to binary labels."""
    if by in df.columns:
        return by
    if by == "sentiment" and "label" in df.columns:
        return "label"
    raise KeyError(f"Column '{by}' not found")


def plot_class_balance(df, label_col="sentiment", name="class_balance"):
    """Plot the number of examples in each sentiment class."""
    label_col = _group_column(df, label_col)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(data=df, x=label_col, ax=ax)
    ax.set(title="Class balance", xlabel="Sentiment class", ylabel="Number of examples")
    fig.tight_layout()
    return _save(fig, name)


def plot_length_distribution(df, text_col, by="sentiment", name="length_distribution"):
    """Plot text length in words, separated by sentiment class."""
    by = _group_column(df, by)
    data = df[[text_col, by]].copy()
    data["word_count"] = data[text_col].fillna("").astype(str).str.split().str.len()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=data, x="word_count", hue=by, bins=20, multiple="layer", ax=ax)
    ax.set(title="Text length distribution", xlabel="Words per text", ylabel="Frequency")
    fig.tight_layout()
    return _save(fig, name)


def plot_top_words(df, by="sentiment", n=15, name="top_words"):
    """Plot the most frequent words for each class using cleaned text when available."""
    by = _group_column(df, by)
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    if text_col not in df.columns:
        raise KeyError("Expected 'clean_text' or 'text' column")
    groups = list(df.groupby(by, dropna=False, sort=True))
    if not groups:
        raise ValueError("Cannot plot top words from an empty DataFrame")
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, max(4, 3.5 * len(groups))), squeeze=False)
    for ax, (group, subset) in zip(axes.flat, groups):
        tokens = (token.lower() for text in subset[text_col].dropna().astype(str)
                  for token in re.findall(r"\b\w+\b", text))
        common = Counter(tokens).most_common(n)
        if common:
            words, counts = zip(*reversed(common))
            ax.barh(words, counts)
        ax.set(title=f"Top words — {group}", xlabel="Frequency", ylabel="Word")
    fig.tight_layout()
    return _save(fig, name)


def plot_wordcloud(texts, name):
    """Generate a word cloud from a string or an iterable of texts."""
    corpus = texts if isinstance(texts, str) else " ".join(str(t) for t in texts if pd.notna(t))
    if not corpus.strip():
        raise ValueError("Word cloud requires nonempty text")
    cloud = WordCloud(width=1000, height=500, background_color="white", random_state=1234).generate(corpus)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(cloud, interpolation="bilinear")
    ax.set_title("Word cloud")
    ax.axis("off")
    fig.tight_layout()
    return _save(fig, name)


def plot_confusion_matrix(y_true, y_pred, labels, name):
    """Plot a confusion matrix for the supplied ordered class labels."""
    labels = list(labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(title="Confusion matrix", xlabel="Predicted label", ylabel="True label")
    fig.tight_layout()
    return _save(fig, name)


def plot_learning_curves(history, name):
    """Plot available training and validation metrics from a Keras History or dict."""
    values = history.history if hasattr(history, "history") else history
    if not values:
        raise ValueError("History has no metrics to plot")
    base_metrics = [key for key in values if not key.startswith("val_")]
    if not base_metrics:
        raise ValueError("History has no training metrics")
    fig, axes = plt.subplots(1, len(base_metrics), figsize=(6 * len(base_metrics), 4), squeeze=False)
    for ax, metric in zip(axes.flat, base_metrics):
        ax.plot(range(1, len(values[metric]) + 1), values[metric], label="Train")
        if f"val_{metric}" in values:
            validation = values[f"val_{metric}"]
            ax.plot(range(1, len(validation) + 1), validation, label="Validation")
        ax.set(title=f"{metric.replace('_', ' ').title()} by epoch", xlabel="Epoch", ylabel=metric)
        ax.legend()
    fig.tight_layout()
    return _save(fig, name)


def plot_model_comparison(df, name="model_comparison"):
    """Compare model scores across train and test splits."""
    if "model" not in df.columns:
        raise KeyError("Expected 'model' column")
    metric = "kappa" if "kappa" in df.columns else "accuracy"
    if metric not in df.columns:
        raise KeyError("Expected 'kappa' or 'accuracy' column")
    fig, ax = plt.subplots(figsize=(9, 5))
    if "split" in df.columns:
        sns.barplot(data=df, x="model", y=metric, hue="split", ax=ax)
    else:
        sns.barplot(data=df, x="model", y=metric, ax=ax)
    ax.set(title=f"Model comparison — {metric}", xlabel="Model", ylabel=metric.title())
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return _save(fig, name)