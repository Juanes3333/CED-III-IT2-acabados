"""Data loading, sentiment mapping, text cleaning, tokenization and splitting.

Implemented so far (IT2-05): load_goemotions, load_sentiment_mapping,
assign_sentiment.
Planned public functions (later commits):
    build_balanced_dataset, clean_text, tokenize, split_dataset,
    build_vectorizer.
"""

import json
from pathlib import Path

import pandas as pd

SEED: int = 1234
ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = ROOT / "data" / "raw"

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
