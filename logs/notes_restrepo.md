# Experiment notes - Restrepo

Entries follow `logs/NOTES_TEMPLATE.md` (added in IT2-04).

## IT2-05: Load GoEmotions and map emotions to sentiment

- **Date:** 2026-11-17
- **Owner:** Restrepo
- **Branch:** `feat/data-pipeline`
- **Commit Reference:** Refs: IT2-05

### 1. Technical Decision / Hypothesis
- Load the three raw TSV files, concatenate them and map each comment to one sentiment group with the official mapping.
- A comment is `positive`, `negative`, `ambiguous` or `neutral` only if all its emotions are in that one group. Any mix of groups is `mixed`. This includes neutral + another emotion.
- `neutral` is not in `sentiment_mapping.json`, so `load_sentiment_mapping` adds it as its own group.

### 2. Context & Rationale
- `MASTER_PROMPT.md` §6.2 steps 1-4 and §7. The rule above is the one that reproduces the statement counts.

### 3. Implementation Details
- Files: `src/preprocessing.py`, `data/raw/` (`train.tsv`, `dev.tsv`, `test.tsv`, `emotions.txt`, `sentiment_mapping.json`), `logs/notes_restrepo.md`.
- Raw files come from `google-research/google-research`, folder `goemotions/data`. Total size is about 4.4 MB, so they are committed (under the 10 MB limit).
- Real format checked before coding: no header, 3 tab-separated columns `text`, `emotion_ids` (comma-separated), `comment_id`. Line counts: train 43,410, dev 5,426, test 5,427 (total 54,263). `emotions.txt` has 28 names, `neutral` is id 27.
- Files are read with `quoting=3` (no quote handling) and `dtype=str`, so comments with quotes or "NA" text are not altered.
- `assign_sentiment` reads `emotions.txt` from the default `RAW_DIR`, because the contract signature has no `raw_dir`.

### 4. Verification & Empirical Evidence
- Command: `python -c "from src.preprocessing import *; ..."` (pandas in a temporary venv).
- Result: 54,263 rows, `comment_id` unique, no missing values.

| Group | Obtained | Statement |
|---|---|---|
| positive | 19,020 | 19,020 |
| negative | 10,217 | 10,217 |
| ambiguous | 4,805 | 4,805 |
| neutral | 16,021 | 16,021 |
| mixed | 4,200 | 4,200 |

- Emotions per comment: 1 -> 45,446; 2 -> 8,124; 3 -> 655; 4 -> 37; 5 -> 1.
- Acceptance criteria checklist:
  - [x] Counts match, no difference to document
  - [x] Functions have docstrings and type hints

### 5. Observations & Teammate Hand-off
- Not done on purpose: deduplication and balanced sampling (IT2-06), cleaning and split (IT2-07).
- `load_emotion_names` is an extra helper, not in the §7 contract.
