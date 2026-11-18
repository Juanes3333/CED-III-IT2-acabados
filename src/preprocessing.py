"""Data loading, sentiment mapping, text cleaning, tokenization and splitting.

Implemented so far: load_goemotions, load_sentiment_mapping,
assign_sentiment (IT2-05), build_balanced_dataset (IT2-06), clean_text,
tokenize, split_dataset, build_vectorizer (IT2-07).
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED: int = 1234
ROOT: Path = Path(__file__).resolve().parents[1]
RAW_DIR: Path = ROOT / "data" / "raw"
PROCESSED_DIR: Path = ROOT / "data" / "processed"
BALANCED_CSV: Path = PROCESSED_DIR / "goemotions_balanced.csv"
TRAIN_CSV: Path = PROCESSED_DIR / "train.csv"
TEST_CSV: Path = PROCESSED_DIR / "test.csv"

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


# Words that flip or weaken sentiment. They are never removed as stopwords.
NEGATIONS: frozenset[str] = frozenset(
    {"no", "not", "nor", "never", "none", "nothing", "nobody", "neither",
     "nowhere", "cannot", "without", "hardly", "barely"}
)

# Common English stopwords (NLTK-style). Written here so no corpus download
# is needed. Negations are subtracted when ``keep_negations`` is True.
_BASE_STOPWORDS: frozenset[str] = frozenset(
    """a about above after again against all am an and any are as at be because
    been before being below between both but by can could did do does doing
    down during each few for from further had has have having he her here hers
    herself him himself his how i if in into is it its itself just me more most
    my myself of off on once only or other our ours ourselves out over own same
    she should so some such than that the their theirs them themselves then
    there these they this those through to too under until up very was we were
    what when where which while who whom why will with would you your yours
    yourself yourselves""".split()
) | NEGATIONS

_NEGATIVE_CONTRACTIONS: tuple[tuple[str, str], ...] = (
    (r"\bwon't\b", "will not"),
    (r"\bcan't\b", "can not"),
    (r"\bain't\b", "is not"),
    (r"n't\b", " not"),
)
_MASK_PATTERN = re.compile(r"\[(?:name|religion)\]")
_CONTRACTION_SUFFIX = re.compile(r"'(?:s|m|re|ve|ll|d)\b")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def clean_text(
    text: str, remove_stopwords: bool = True, keep_negations: bool = True
) -> str:
    """Clean one comment for modeling.

    Steps: lowercase; drop the masked tokens ``[NAME]`` and ``[RELIGION]``
    (they hide a person or group and carry no sentiment); turn negative
    contractions into ``not`` (``don't`` -> ``do not``) so negation survives
    punctuation removal; drop other contraction suffixes; remove punctuation;
    optionally remove stopwords while keeping negations.

    Args:
        text: Raw comment.
        remove_stopwords: Remove common English stopwords.
        keep_negations: Keep words like ``not`` and ``never`` even when
            ``remove_stopwords`` is True.

    Returns:
        Cleaned text with tokens separated by one space (may be empty).
    """
    text = text.lower().replace("\u2019", "'")
    text = _MASK_PATTERN.sub(" ", text)
    for pattern, replacement in _NEGATIVE_CONTRACTIONS:
        text = re.sub(pattern, replacement, text)
    text = _CONTRACTION_SUFFIX.sub("", text)
    tokens = _TOKEN_PATTERN.findall(text)
    if remove_stopwords:
        stopwords = _BASE_STOPWORDS - NEGATIONS if keep_negations else _BASE_STOPWORDS
        tokens = [t for t in tokens if t not in stopwords]
    return " ".join(tokens)


def tokenize(text: str) -> list[str]:
    """Split a cleaned text into tokens.

    Args:
        text: Output of :func:`clean_text`.

    Returns:
        List of tokens (empty list for empty text).
    """
    return text.split()


def split_dataset(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train and test, stratified by ``label`` and seeded.

    Do this before fitting any vectorizer so the test set stays unseen.

    Args:
        df: Balanced dataset with a ``label`` column.
        test_size: Fraction of rows for the test set.
        seed: Random seed.

    Returns:
        ``(train, test)`` DataFrames with a fresh index.
    """
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_vectorizer(texts: pd.Series, max_tokens: int, max_len: int):
    """Build a Keras ``TextVectorization`` layer adapted on ``texts``.

    Pass the **train** texts only, so the vocabulary never sees the test set.

    Args:
        texts: Training texts (usually the ``clean_text`` column).
        max_tokens: Maximum vocabulary size.
        max_len: Output sequence length (padding / truncation).

    Returns:
        An adapted ``keras.layers.TextVectorization`` layer.
    """
    from tensorflow import keras  # imported here: only the models need it

    vectorizer = keras.layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=max_len,
    )
    vectorizer.adapt(texts.to_numpy())
    return vectorizer


def _sanity_checks(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Small assert-based checks for cleaning and the split."""
    assert "not" in clean_text("This is not good").split()
    assert clean_text("I don't like it") == "not like"
    assert clean_text("Never again!") == "never"
    assert clean_text("Thanks [NAME]!") == "thanks"
    assert "not" not in clean_text("not good", keep_negations=False).split()
    assert tokenize("not good") == ["not", "good"] and tokenize("") == []
    assert len(train) == 800 and len(test) == 200
    for part in (train, test):
        assert part["label"].value_counts().to_dict() == {0: len(part) // 2, 1: len(part) // 2}
    assert not set(train["comment_id"]) & set(test["comment_id"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    full = assign_sentiment(load_goemotions(), load_sentiment_mapping())
    out = save_balanced_dataset(build_balanced_dataset(full))
    logger.info("Saved %s", out)

    balanced = pd.read_csv(BALANCED_CSV)
    balanced["clean_text"] = balanced["text"].apply(clean_text)
    train_df, test_df = split_dataset(balanced)
    _sanity_checks(train_df, test_df)
    columns = ["comment_id", "text", "clean_text", "label"]
    for frame, path in ((train_df, TRAIN_CSV), (test_df, TEST_CSV)):
        frame[columns].to_csv(path, index=False, encoding="utf-8", lineterminator="\n")
        logger.info("Saved %s (%d rows)", path, len(frame))
    logger.info("Empty clean_text rows: %d", int((balanced["clean_text"] == "").sum()))
