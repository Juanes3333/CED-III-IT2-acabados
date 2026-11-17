"""Data loading, sentiment mapping, text cleaning, tokenization and splitting.

Implemented so far: load_goemotions, load_sentiment_mapping,
assign_sentiment (IT2-05), build_balanced_dataset (IT2-06).
Planned public functions (later commits):
    clean_text, tokenize, split_dataset, build_vectorizer.
"""

import json
import logging
from pathlib import Path

import pandas as pd

SEED: int = 1234
ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = ROOT / "data" / "raw"
PROCESSED_DIR: Path = ROOT / "data" / "processed"
BALANCED_CSV: Path = PROCESSED_DIR / "goemotions_balanced.csv"

logger = logging.getLogger(__name__)

SPLIT_FILES: tuple[str, ...] = ("train.tsv", "dev.tsv", "test.tsv")
NEUTRAL_GROUP: str = "neutral"
MIXED_GROUP: str = "mixed"


def load_emotion_names(raw_dir: Path = RAW_DIR) -> list[str]:
    """Read the 28 emotion names from ``emotions.txt`` (index = emotion id).

    Args:
        raw_dir: Folder with the raw GoEmotions files.

    Returns:
        List of emotion names; position ``i`` is the name of id ``i``.
    """
    lines = (Path(raw_dir) / "emotions.txt").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def load_goemotions(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and concatenate ``train.tsv``, ``dev.tsv`` and ``test.tsv``.

    The raw files have no header and three tab-separated columns:
    text, comma-separated emotion ids and the Reddit comment id.

    Args:
        raw_dir: Folder with the raw GoEmotions files.

    Returns:
        DataFrame with columns ``comment_id``, ``text``, ``emotion_ids``
        (``emotion_ids`` is a list of ints). One row per comment.
    """
    frames = []
    for name in SPLIT_FILES:
        frame = pd.read_csv(
            Path(raw_dir) / name,
            sep="\t",
            header=None,
            names=["text", "emotion_ids", "comment_id"],
            dtype=str,
            keep_default_na=False,
            quoting=3,  # csv.QUOTE_NONE: comments may contain quote characters
        )
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["emotion_ids"] = df["emotion_ids"].apply(
        lambda ids: [int(i) for i in ids.split(",")]
    )
    return df[["comment_id", "text", "emotion_ids"]]


def load_sentiment_mapping(raw_dir: Path = RAW_DIR) -> dict[str, str]:
    """Load the official sentiment grouping as ``emotion name -> group``.

    ``sentiment_mapping.json`` lists the positive, negative and ambiguous
    emotions. ``neutral`` is not in the file, so it is added as its own group.

    Args:
        raw_dir: Folder with the raw GoEmotions files.

    Returns:
        Dict that maps each of the 28 emotion names to ``positive``,
        ``negative``, ``ambiguous`` or ``neutral``.
    """
    path = Path(raw_dir) / "sentiment_mapping.json"
    groups = json.loads(path.read_text(encoding="utf-8"))
    mapping = {emotion: group for group, names in groups.items() for emotion in names}
    mapping["neutral"] = NEUTRAL_GROUP
    return mapping


def assign_sentiment(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Add the emotion names and the sentiment group of each comment.

    A comment gets ``positive``, ``negative``, ``ambiguous`` or ``neutral``
    only if all its emotions belong to that single group. If its emotions
    come from more than one group (neutral + another emotion included), it
    is ``mixed``.

    Args:
        df: Output of :func:`load_goemotions`.
        mapping: Output of :func:`load_sentiment_mapping`.

    Returns:
        Copy of ``df`` with two new columns: ``emotions`` (list of names)
        and ``sentiment`` (group name).
    """
    names = load_emotion_names()
    out = df.copy()
    out["emotions"] = out["emotion_ids"].apply(lambda ids: [names[i] for i in ids])

    def group_of(emotions: list[str]) -> str:
        groups = {mapping[e] for e in emotions}
        return groups.pop() if len(groups) == 1 else MIXED_GROUP

    out["sentiment"] = out["emotions"].apply(group_of)
    return out


def normalize_text(text: str) -> str:
    """Normalize a comment only to compare duplicates (not for modeling).

    Lowercases, strips the ends and collapses repeated whitespace.

    Args:
        text: Raw comment text.

    Returns:
        Normalized text.
    """
    return " ".join(text.lower().split())


def build_balanced_dataset(
    df: pd.DataFrame, n_per_class: int = 500, seed: int = SEED
) -> pd.DataFrame:
    """Build the balanced binary dataset (positive vs negative).

    Steps: keep only pure ``positive`` and ``negative`` comments, remove
    duplicated comments (normalized text), then sample ``n_per_class``
    comments per class with ``seed``. If the same normalized text appears
    with both labels, all its copies are dropped because its label cannot be
    trusted. Otherwise the first copy (by ``comment_id``) is kept. The number
    of removed duplicates and conflicts is logged.

    Args:
        df: Output of :func:`assign_sentiment`.
        n_per_class: Number of comments to sample for each class.
        seed: Random seed for sampling and shuffling.

    Returns:
        DataFrame with columns ``comment_id``, ``text``, ``sentiment`` and
        ``label`` (1 = positive, 0 = negative), shuffled with ``seed``.

    Raises:
        ValueError: If a class has fewer than ``n_per_class`` comments.
    """
    pure = df[df["sentiment"].isin(["positive", "negative"])].copy()
    pure = pure.sort_values("comment_id", kind="stable").reset_index(drop=True)
    pure["_key"] = pure["text"].apply(normalize_text)

    n_labels = pure.groupby("_key")["sentiment"].transform("nunique")
    conflict = n_labels > 1
    n_conflict_texts = pure.loc[conflict, "_key"].nunique()
    pure = pure[~conflict]

    n_before = len(pure)
    pure = pure.drop_duplicates(subset="_key", keep="first")
    logger.info("Pure positive/negative comments: %d", len(df[df["sentiment"].isin(["positive", "negative"])]))
    logger.info(
        "Texts with conflicting labels (all copies dropped): %d (%d rows)",
        n_conflict_texts,
        int(conflict.sum()),
    )
    logger.info("Duplicate rows removed: %d", n_before - len(pure))

    parts = []
    for sentiment in ("positive", "negative"):
        group = pure[pure["sentiment"] == sentiment]
        if len(group) < n_per_class:
            raise ValueError(
                f"Only {len(group)} {sentiment} comments, need {n_per_class}"
            )
        parts.append(group.sample(n=n_per_class, random_state=seed))

    balanced = pd.concat(parts).sample(frac=1.0, random_state=seed)
    balanced["label"] = (balanced["sentiment"] == "positive").astype(int)
    return balanced[["comment_id", "text", "sentiment", "label"]].reset_index(drop=True)


def save_balanced_dataset(balanced: pd.DataFrame, path: Path = BALANCED_CSV) -> Path:
    """Write the balanced dataset as UTF-8 CSV with Unix line endings.

    Args:
        balanced: Output of :func:`build_balanced_dataset`.
        path: Destination file.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    balanced.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    full = assign_sentiment(load_goemotions(), load_sentiment_mapping())
    out = save_balanced_dataset(build_balanced_dataset(full))
    logger.info("Saved %s", out)
