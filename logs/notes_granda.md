# Experiment notes - Granda

Entries follow `logs/NOTES_TEMPLATE.md` (added in IT2-04).

## IT2-03 - Documentation and Dataset Scaffolding

**Date:** 2026-11-17<br>
**Owner:** Samuel Granda<br>
**Branch:** `docs/scaffold`<br>
**Commit ID:** IT2-03

### Summary of Work
- Created comprehensive `README.md` skeleton following course requirements and repo tree from `MASTER_PROMPT.md` §2.
  - Included project context, team roles, environment setup instructions (conda and pip), execution workflow order, experimental guidelines (fixed seed `1234`, no data leakage, metrics), GenAI Level 3 disclosure, and placeholders `TODO(IT2-32)`.
- Created `data/README.md` with:
  - Complete academic citation of Demszky et al. (ACL 2020), DOI, arXiv, Kaggle link, and official Google Research mapping JSON.
  - Emotion mapping taxonomy (positive, negative, ambiguous, neutral) and exact expected counts (positive: 19,020, negative: 10,217, ambiguous: 4,805, neutral: 16,021, mixed: 4,200; total: 54,263).
  - Target subset specification: 500 positive + 500 negative comments, deduplication, 80/20 train/test split.
  - Schema documentation for processed files (`goemotions_balanced.csv`, `train.csv`, `test.csv`) with placeholders for preprocessing decisions (`TODO(IT2-07)`).

### Verification
- Both Markdown files render properly.
- URLs and citations verified against `MASTER_PROMPT.md`.
