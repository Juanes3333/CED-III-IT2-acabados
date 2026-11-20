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

## IT2-06: Build balanced 500/500 sentiment dataset

- **Date:** 2026-11-17
- **Owner:** Restrepo
- **Branch:** `feat/data-pipeline`
- **Commit Reference:** Refs: IT2-06

### 1. Technical Decision / Hypothesis
- Keep only pure positive/negative comments, remove duplicates, sample 500 + 500 with `SEED = 1234`.
- Duplicates are compared with normalized text (lowercase, trimmed, collapsed spaces). The first copy by `comment_id` is kept.
- If one normalized text appears with both labels, all its copies are dropped, because the label cannot be trusted.

### 2. Context & Rationale
- `MASTER_PROMPT.md` §6.2 steps 5-6. Sorting by `comment_id` before deduplication and sampling makes the result independent of file order.

### 3. Implementation Details
- Files: `src/preprocessing.py`, `data/processed/goemotions_balanced.csv`, `logs/notes_restrepo.md`.
- New functions: `build_balanced_dataset` (contract §7), plus helpers `normalize_text` and `save_balanced_dataset`. Running `python -m src.preprocessing` from the repo root writes the CSV.
- CSV columns: `comment_id, text, sentiment, label` (1 = positive, 0 = negative), shuffled with the seed, LF line endings.

### 4. Verification & Empirical Evidence
- Command: `python -m src.preprocessing` (twice), then `md5sum` and a pandas check.
- Pure positive/negative comments: 29,237 (19,020 + 10,217).
- Conflicting labels: 1 text, 3 rows, all dropped.
- Duplicate rows removed (after conflicts): 213.
- Output: 1,000 rows, 500 positive / 500 negative, 0 duplicate normalized texts, `comment_id` unique.
- Both runs gave the same MD5 `9912473c1d6d464bf5da54da746153aa`.
- Acceptance criteria checklist:
  - [x] 1,000 rows, 500/500
  - [x] No duplicate text
  - [x] Byte-identical CSV on two runs

### 5. Observations & Teammate Hand-off
- Not done on purpose: cleaning, tokenization and split (IT2-07).
- Duplicates were checked only among pure positive/negative comments, not against neutral, ambiguous or mixed ones.

## IT2-07: Text cleaning, tokenization and split

- **Date:** 2026-11-17
- **Owner:** Restrepo
- **Branch:** `feat/data-pipeline`
- **Commit Reference:** Refs: IT2-07

### 1. Technical Decision / Hypothesis
- `clean_text`: lowercase, drop `[NAME]` / `[RELIGION]`, turn negative contractions into `not`, remove punctuation, remove stopwords but **keep negations** (`no, not, nor, never, none, nothing, nobody, neither, nowhere, cannot, without, hardly, barely`).
- `[NAME]` / `[RELIGION]` are dropped, not kept as tokens: they hide a person or group, carry no sentiment, and `[NAME]` appears 143 times in the 1,000 comments, so it would become a noisy frequent word.
- Stratified 80/20 split with `random_state=SEED`, done before any vectorizer is fitted.

### 2. Context & Rationale
- `MASTER_PROMPT.md` §6.2 steps 7-8 and §4.2 (the slides remove negations with the standard stopword list).
- Negative contractions are expanded before removing punctuation. Without this, `don't` becomes `don t` and the negation is lost.
- The stopword list is written in the code (NLTK-style) so nobody needs to download an NLTK corpus.

### 3. Implementation Details
- Files: `src/preprocessing.py`, `data/processed/train.csv`, `data/processed/test.csv`, `logs/notes_restrepo.md`.
- New functions: `clean_text`, `tokenize`, `split_dataset`, `build_vectorizer` (contract §7). Helper: `_sanity_checks`. Running `python -m src.preprocessing` rebuilds the three CSVs.
- `build_vectorizer` uses Keras `TextVectorization` (`output_mode="int"`, `output_sequence_length=max_len`) and imports TensorFlow inside the function. Pass only the train texts.
- Processed CSV columns: `comment_id, text, clean_text, label`.

### 4. Verification & Empirical Evidence
- Command: `python -m src.preprocessing` (twice) and a pandas check.
- 800 train rows (400/400) and 200 test rows (100/100). No `comment_id` in both sets. `assert` sanity checks pass.
- Same MD5 on both runs: `train.csv` `2e701a4830c5f77a39793494fb786211`, `test.csv` `7ec72f469b2e7940e50bd3c5286352ca`.
- Negations kept: `"This is not good"` -> `not good`; `"I don't like it"` -> `not like`; `"Isn’t that great?"` -> `not great`. 214 comments have a negation in the text and 214 keep `not` in `clean_text`.
- No empty `clean_text` rows. Tokens per cleaned comment: median 6, 95th percentile 13, 99th percentile 15, max 17 (input for `max_len` in the EDA commit).
- Acceptance criteria checklist:
  - [x] 800 train / 200 test, 50/50 in each
  - [x] Negations preserved
  - [x] Sanity checks pass
  - [ ] `build_vectorizer` was **not run**: TensorFlow is not installed on this machine (only a temporary venv with pandas and scikit-learn). It must be run in the `environment.yml` environment.

### 5. Observations & Teammate Hand-off
- Emojis are removed by the cleaning (e.g. `🙄`) and `10/10` becomes `10 10`. This loses a little sentiment signal; the EDA can test keeping `!` and `?` as an ablation (§6.2).
- Contractions like `i'm` become `i`, which is then removed as a stopword.
- Not done on purpose: `eda_summary.json` and the `max_len` / `max_words` values (EDA commits), and the with/without-negation ablation.

## IT2-16: Dense neural network on TF-IDF features

- **Date:** 2026-11-19
- **Owner:** Restrepo
- **Branch:** `feat/dense-nn`
- **Commit Reference:** Refs: IT2-16

### 1. Technical Decision / Hypothesis
- `build_dense_pipeline(**params)` returns `Pipeline([("tfidf", TfidfVectorizer), ("mlp", MLPClassifier)])`.
- One pipeline means the TF-IDF vocabulary is learned only from the training rows of each fit or CV fold. No test or validation text leaks into the features.
- Defaults are a small starting point for IT2-17 to tune: TF-IDF with unigrams + bigrams, `min_df=2`, `sublinear_tf=True`; MLP with one hidden layer of 64 ReLU units, `alpha=1e-3`, `max_iter=300`.
- `early_stopping=True` uses a 10% validation slice taken inside the training data, so the test set is not involved.
- `random_state=SEED` (1234) in the MLP, so runs are repeatable.

### 2. Context & Rationale
- `MASTER_PROMPT.md` §7 contract and `COMMITS.md` IT2-16 (slide 23 p.28-30: pipeline prevents leakage).
- Parameters are set by name with the sklearn `step__param` syntax (`mlp__hidden_layer_sizes`, `tfidf__max_features`, ...), so the same names work in the `GridSearchCV` grid of IT2-17.
- sklearn is imported inside the function, so importing `src.models` does not need sklearn or TensorFlow (the RNN/LSTM/Transformer sections will add their own imports).

### 3. Implementation Details
- Files: `src/models.py` (only the `# --- Dense ---` section), `logs/notes_restrepo.md`, `prompts/log_restrepo.md`.
- Added `SEED = 1234` at the top of the Dense section (the file had no `SEED`). Later sections can reuse it.

### 4. Verification & Empirical Evidence
- Environment: temporary venv with Python 3.14, scikit-learn 1.9.1, pandas 3.1.1. This is **not** the pinned `environment.yml` (scikit-learn 1.4.2); the code uses only stable API, but IT2-17 and IT2-18 should be run in the pinned environment.
- Toy sample (30 short texts): the pipeline fits and predicts without errors. The toy predictions are not meaningful (too little data); only the fit was checked.
- Parameters by name: `mlp__hidden_layer_sizes=(32, 16)`, `mlp__alpha=0.01`, `tfidf__max_features=500` are read back with `get_params()`; `set_params(mlp__hidden_layer_sizes=(8,))` works; an invalid name raises `ValueError`.
- Fit on `data/processed/train.csv` (800 rows, `clean_text`): vocabulary 786 terms, train accuracy 0.929. The test file was not read.
- Acceptance criteria checklist:
  - [x] Fits on a toy sample
  - [x] Params settable by name (`mlp__hidden_layer_sizes`, ...)

### 5. Observations & Teammate Hand-off
- Not done on purpose: `run_dense_grid` and the grid search (IT2-17), the notebook and metrics (IT2-18).
- Train accuracy of 0.929 is a sanity number only. Honest performance comes from CV and the single test evaluation later.
- Observation: the `MASTER_PROMPT.md` §7 contract says `build_dense_pipeline(**params)`; the defaults live in the function, not in the signature.

## IT2-17: Tune dense network with GridSearchCV on kappa

- **Date:** 2026-11-19
- **Owner:** Restrepo
- **Branch:** `feat/dense-nn`
- **Commit Reference:** Refs: IT2-17

### 1. Technical Decision / Hypothesis
- `run_dense_grid(X_train, y_train, param_grid, cv=5)` runs `GridSearchCV` on the pipeline from `build_dense_pipeline()`.
- Scoring is `make_scorer(cohen_kappa_score)`. Accuracy would hide a bias towards one class; kappa corrects for chance agreement, and the course asks for it.
- Folds are `StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)`: both classes keep the same proportion in every fold and the folds are repeatable.
- `refit=True`, so `best_estimator_` is already refitted on the whole training split and IT2-18 can use it directly.
- `return_train_score=True` is added so the CSV shows the train-vs-validation gap of each candidate (overfitting evidence for the report).
- The function only receives training texts and labels. The test file is never read during tuning.

### 2. Context & Rationale
- `MASTER_PROMPT.md` §6 (Dense NN row) and §4.2 (slide 18, p.48 passes the bare metric function as `scoring`; that is not a scorer).
- `DENSE_PARAM_GRID` in `src/train.py` holds the grid of §6: `hidden_layer_sizes` {(32,), (64,), (128,), (64, 32)} x `activation` {relu, tanh} x `alpha` {1e-4, 1e-3, 1e-2} x `learning_rate_init` {1e-3, 1e-2} = 48 candidates, 240 fits with 5 folds. Keys use the `mlp__` prefix of IT2-16.
- Two optional keyword-only arguments were added after the contract signature of §7: `output_dir` (default `outputs/metrics`, used to test without touching the real outputs) and `n_jobs`. Calls that follow the §7 signature still work.
- sklearn and `src.models` are imported inside the function, so `import src.train` still needs only NumPy and pandas.

### 3. Implementation Details
- Files: `src/train.py` (new section `# --- Dense (sklearn) grid search ---`), `logs/notes_restrepo.md`, `prompts/log_restrepo.md`.
- Outputs written by the function:
  - `outputs/metrics/dense_grid_results.csv`: all 48 candidates sorted by rank, with `param_*` columns, per-fold and mean validation kappa, train kappa and fit time. It uses the existing `save_grid_results`. The `params` column is a JSON string.
  - `outputs/metrics/dense_best_params.json`: `best_params`, `best_cv_kappa`, scoring name, folds, number of candidates, rows and seed. Rebuild the model with `build_dense_pipeline(**json.load(f)["best_params"])`. Tuples are saved as lists (`[64]`); sklearn accepts them.
  - `logs/training_logs_dense.txt` through `get_logger("dense")`: start line, best score and params, seconds, saved paths.

### 4. Verification & Empirical Evidence
- Environment: temporary Python 3.11.18 environment with the pinned versions (scikit-learn 1.4.2, pandas 2.2.2, numpy 1.26.4). IT2-16 was first checked with scikit-learn 1.9.1; it also fits and sets parameters by name with 1.4.2.
- Toy run (60 texts, 4 candidates, 3 folds, output in a scratch folder): returns a `GridSearchCV`, writes the CSV, the JSON and the log.
- Real run on `data/processed/train.csv` (800 rows, 400 per class), 48 candidates x 5 folds, 12.5 seconds with `n_jobs=-1`.
- Best candidate: `relu`, `alpha=0.01`, `hidden_layer_sizes=(64,)`, `learning_rate_init=0.01`, mean CV kappa **0.470** (std 0.040), mean train kappa 0.834. Validation kappa across all candidates goes from 0.31 to 0.47.
- Rebuilding the pipeline from the saved JSON and fitting it on the train split works (train accuracy 0.959).
- Acceptance criteria checklist:
  - [x] Grid runs on the train split only (the function never receives test data)
  - [x] Best params saved (`dense_best_params.json`) and all CV results saved (`dense_grid_results.csv`)
  - [x] The scorer fix is written in `prompts/log_restrepo.md` as a critique of slide 18

### 5. Observations & Teammate Hand-off
- 22 of the 48 candidates are within one standard deviation of the best mean kappa (0.040). The "best" configuration is not clearly better than its neighbours; say this in the report instead of presenting it as a sharp winner.
- The gap between train kappa (0.83) and validation kappa (0.47) shows that the network overfits the 800 training rows. A stronger `alpha` or a smaller vocabulary (`tfidf__max_features`) could be explored later, but they are not part of this grid.
- Not done on purpose: the train and test metrics, the confusion matrix, the top TF-IDF weights, the notebook and the saved model (IT2-18). The test set has not been used.
- The run created `logs/training_logs_dense.txt`, `outputs/metrics/dense_grid_results.csv` and `outputs/metrics/dense_best_params.json`. They are not in the "Files" line of IT2-17 in `COMMITS.md`; the team should decide whether to commit them here or with IT2-18.
- Reproducibility: re-running the whole grid single-threaded (`n_jobs=1`) gave exactly the same ranking and the same mean kappa (maximum difference 0.0) as the `n_jobs=-1` run. Results on another machine or library version may still differ.

## IT2-18: Dense network section in the model notebook

- **Date:** 2026-11-19
- **Owner:** Restrepo
- **Branch:** `feat/dense-nn`
- **Commit Reference:** Refs: IT2-18

### 1. Technical Decision / Hypothesis
- `notebooks/03_dense_rnn_lstm.ipynb` is created with: a header (objective, outputs, step order, owners), a shared setup (imports, seed, load train and test), the full Dense section, and three empty headed sections `## RNN`, `## LSTM` and `## Comparison` for Varela and Granda.
- The Dense section follows the professor's step order (slide 18): importing libraries, loading, splitting, building, hyperparameter fine tuning, best configuration, training, prediction on train, prediction on test, metrics. Titles are questions and there is one step per cell.
- The tuning cell calls `run_dense_grid` (IT2-17), so the notebook runs from top to bottom from a fresh clone and does not depend on files that are not in the repository.
- The test set is used once, after the configuration is fixed. The training step rebuilds the pipeline with `best_params_` and checks that it predicts the same as `grid.best_estimator_`.

### 2. Context & Rationale
- Report both splits with the same metrics (rubric: overfitting check) and an insight text after every output (`MASTER_PROMPT.md` §4.1 and §6).
- "Top TF-IDF weights": an `MLPClassifier` has no single weight per word. We use a one-word probe: each term alone is a unit TF-IDF vector (rows are L2-normalised) and we read the log-odds of the positive class. Plain probabilities saturate near 0 and 1 (the first version of the figure showed all bars full), so the figure uses log-odds. The notebook says this is a probe and not a linear coefficient.
- The shared setup seeds Python and NumPy but does not call `src.train.set_seed`, because that function needs TensorFlow and the Dense section does not. A comment tells the RNN and LSTM authors to call it in their sections.
- Honest comment on the 200-row test set: the notebook prints a rough 95% interval for the accuracy (normal approximation). The bootstrap intervals belong to IT2-24.

### 3. Implementation Details
- Files committed: `notebooks/03_dense_rnn_lstm.ipynb`, `outputs/metrics/dense_metrics.csv`, `outputs/figures/dense_confusion_matrix.png`, `outputs/figures/dense_top_terms.png`, `outputs/saved_models/dense_pipeline.joblib` (1.3 MB, small enough to commit), `logs/notes_restrepo.md`, `prompts/log_restrepo.md`.
- Helpers used from the project: `evaluate_train_test`, `save_metrics`, `confusion_matrix_data`, `plot_confusion_matrix`, `build_dense_pipeline`, `run_dense_grid`, `DENSE_PARAM_GRID`. No source file was changed in this commit.
- Not committed on purpose (local outputs of the tuning cell): `outputs/metrics/dense_grid_results.csv`, `outputs/metrics/dense_best_params.json`, `logs/training_logs_dense.txt`. The team decided they are not part of the commit; running the notebook regenerates them.

### 4. Verification & Empirical Evidence
- Environment: temporary Python 3.11.18 with scikit-learn 1.4.2, pandas 2.2.2, numpy 1.26.4, joblib 1.4.2, matplotlib 3.8.4, seaborn 0.13.2, wordcloud 1.9.3, ipykernel 6.29.4 and nbconvert 7.16.4 (versions of `requirements.txt`). It was not the conda environment of `environment.yml`, and TensorFlow was not installed (this section does not need it).
- `python -m nbconvert --to notebook --execute --inplace notebooks/03_dense_rnn_lstm.ipynb` runs top to bottom with no errors (42 cells), and a second run gives the same numbers.
- `dense_metrics.csv` has a train row and a test row with the five metrics (plus macro F1):
  - train: accuracy 0.959, precision 0.969, recall 0.948, F1 0.958, kappa 0.918
  - test: accuracy 0.720, precision 0.724, recall 0.710, F1 0.717, kappa 0.440
- Baseline comparison: best baseline test accuracy is 0.515 (`dummy_stratified`) and the dense test accuracy is 0.720. The notebook asserts this.
- Test confusion matrix: tn 73, fp 27, fn 29, tp 71.
- CV kappa of the best configuration 0.470 (std 0.040) against test kappa 0.440.
- The saved pipeline reloads and gives the same test predictions. The notebook has no personal paths in its outputs.
- Acceptance criteria checklist:
  - [x] Notebook runs top to bottom
  - [x] Metrics CSV has train and test rows
  - [x] Accuracy above the baseline (0.720 vs 0.515)

### 5. Observations & Teammate Hand-off
- **Varela and Granda:** write only inside `## RNN`, `## LSTM` and `## Comparison`. Do not edit the setup cell; if you need another import, add it at the top of your own section. The tuning cell takes about 12 to 15 seconds.
- Overfitting is large (train accuracy 0.96, test 0.72). It is a result to explain in the report, not a bug.
- 22 of 48 tuned candidates are within one standard deviation of the best, and the test accuracy has a rough interval of about +/- 0.06. The final comparison must not claim a winner when intervals overlap.
- The one-word probe contains some terms that are not obviously emotional (`trust`, `show`, `may`, `mind`, `wish`). They probably come from the small sample; the final error analysis can look at them.
- Merge M8 comes after review of the three commits IT2-16, IT2-17 and IT2-18 (PR from `feat/dense-nn`).
