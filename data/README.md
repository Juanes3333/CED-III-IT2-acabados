# GoEmotions Dataset & Preprocessing Pipeline

This document describes the dataset origin, citation, label taxonomy, expected class distributions, and the data preparation pipeline used in this project.

---

## 1. Dataset Origin & Citation

The **GoEmotions** dataset is a corpus of 58k English Reddit comments labeled with 27 emotion categories plus Neutral, curated to study fine-grained emotion classification.

- **Primary Citation:**
  > Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)*, pp. 4040–4054. DOI: [10.18653/v1/2020.acl-main.372](https://doi.org/10.18653/v1/2020.acl-main.372). arXiv: [2005.00547](https://arxiv.org/abs/2005.00547).

- **Data Sources:**
  - Kaggle repository: [`debarshichanda/goemotions`](https://www.kaggle.com/datasets/debarshichanda/goemotions)
  - Official Google Research repository & sentiment mapping: [`google-research/goemotions`](https://github.com/google-research/google-research/tree/master/goemotions)
  - Mapping JSON: [`sentiment_mapping.json`](https://raw.githubusercontent.com/google-research/google-research/master/goemotions/data/sentiment_mapping.json)

---

## 2. Raw Data Format & Taxonomy

The filtered GoEmotions release consists of `train.tsv`, `dev.tsv`, and `test.tsv`, totaling **54,263 comments** where at least 2 independent annotators agreed on labels.

- **Structure of Raw TSV Files:**
  - Tab-separated values with no header.
  - Column 1: `text` (string, Reddit comment text). Comments are short ($\le 30$ tokens, average $\approx 13$ words). Anonymized entities appear as `[NAME]` and `[RELIGION]` mask tokens.
  - Column 2: `emotion_ids` (comma-separated integers from 0 to 27).
  - Column 3: `comment_id` (Reddit identifier string).
  - Label dictionary: `emotions.txt` lists all 28 classes (index 27 denotes `neutral`).

### Official Sentiment Grouping

Using `sentiment_mapping.json`, the 28 fine-grained emotions map into four primary categories:

| Sentiment Group | Included Emotion Categories |
|---|---|
| **Positive** (12) | `admiration`, `amusement`, `approval`, `caring`, `desire`, `excitement`, `gratitude`, `joy`, `love`, `optimism`, `pride`, `relief` |
| **Negative** (11) | `anger`, `annoyance`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `fear`, `grief`, `nervousness`, `remorse`, `sadness` |
| **Ambiguous** (4) | `confusion`, `curiosity`, `realization`, `surprise` |
| **Neutral** (1) | `neutral` (explicit absence of emotion) |

### Expected Counts on Filtered GoEmotions (Sanity Check)

When aggregating all 54,263 comments across `train.tsv`, `dev.tsv`, and `test.tsv` and assigning high-level sentiment based on emotion consensus:

| Category | Definition | Expected Comments |
|---|---|---|
| **Positive** | All labeled emotions belong to the positive category | 19,020 |
| **Negative** | All labeled emotions belong to the negative category | 10,217 |
| **Ambiguous** | All labeled emotions belong to the ambiguous category | 4,805 |
| **Neutral** | Single label of neutral | 16,021 |
| **Mixed** | Labeled with emotions spanning across multiple categories | 4,200 |
| **Total** | Full filtered GoEmotions corpus | **54,263** |

---

## 3. Sampling and Balancing Protocol

To construct the target dataset for binary sentiment classification:

1. **Filtering:** Retain only comments classified as pure **positive** or pure **negative** (drop neutral, ambiguous, and mixed-emotion comments).
2. **Deduplication:** Normalize text (lowercase, collapse repeated whitespace, strip whitespace) and remove duplicate comments, ensuring no conflicting sentiment labels exist.
3. **Balanced Subsampling:** Randomly draw exactly **500 positive** and **500 negative** comments using fixed random seed `SEED = 1234`.
4. **Stratified Split:** Partition the balanced 1,000 samples into an 80% training set (800 samples: 400 positive / 400 negative) and a 20% test set (200 samples: 100 positive / 100 negative) using `stratify=y` and `random_state=1234`.

---

## 4. Processed Data Files

Processed artifacts are stored in `data/processed/`:

- `data/processed/goemotions_balanced.csv`:
  - Exactly 1,000 rows (500 positive, 500 negative).
  - Columns: `comment_id`, `text`, `sentiment` (`positive` / `negative`), `label` (`1` / `0`).
- `data/processed/train.csv`:
  - 800 rows (400 positive, 400 negative).
  - Columns: `comment_id`, `text`, `clean_text`, `label`.
- `data/processed/test.csv`:
  - 200 rows (100 positive, 100 negative).
  - Columns: `comment_id`, `text`, `clean_text`, `label`.

<!-- TODO(IT2-07): Document finalized text cleaning choices (handling of [NAME]/[RELIGION] tokens, punctuation rules, and negation-preserving stopword list) -->
<!-- TODO(IT2-32): Add final vocabulary size and max_len statistics generated during EDA -->
