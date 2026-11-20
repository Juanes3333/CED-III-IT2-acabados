# Varela - Experiment and Technical Notes

## IT2-02 - Environment Configuration

**Date:** 2026-11-17

**Owner:** Daniel Varela

**GitHub user:** Danielvarelavictoria

### 1. Local environment

- Operating system: Windows
- System Python version: 3.14.7
- Project Python version: 3.11.18
- Environment manager: Conda
- Virtual environment name: goemotions-it2

Python 3.14.7 is installed on the local machine.
However, Python 3.11 was selected for the project
to improve compatibility with TensorFlow.

### 2. Deep Learning framework decision

TensorFlow and Keras were selected as the main
deep learning frameworks.

Pinned versions:

- TensorFlow: 2.15.1
- Keras: 2.15.0

These frameworks will be used for the Dense Neural
Network, Vanilla RNN and LSTM architectures.

### 3. Transformer decision

The initial approach is to use the TensorFlow
integration provided by Hugging Face Transformers.

Pinned version:

- Transformers: 4.38.2

The TensorFlow integration was verified successfully
using TFAutoModelForSequenceClassification.

### 4. NLP dependencies

NLTK was selected for text preprocessing.

Required NLTK resources:

- punkt
- stopwords

Both resources were downloaded successfully.

### 5. Reproducibility

The project will use the following random seed:

- SEED = 1234

All random operations should use this seed whenever
the corresponding library supports it.

### 6. Installation and validation

The Conda environment was created successfully
using environment.yml.

The dependencies in requirements.txt were installed
successfully. PyArrow was pinned to version 15.0.2
to maintain compatibility with NumPy 1.26.4.

Validation checklist:

- [x] Create the Conda environment.
- [x] Install requirements.txt.
- [x] Verify Python version (3.11.18).
- [x] Import NumPy and Pandas.
- [x] Import Scikit-learn.
- [x] Import NLTK.
- [x] Import TensorFlow and Keras.
- [x] Import Transformers and Datasets.
- [x] Import visualization libraries.
- [x] Verify Jupyter and nbconvert dependencies through installation.
- [x] Download NLTK resources (punkt and stopwords).
- [x] Verify TensorFlow Transformer integration.
- [x] Run pip check (no broken requirements found).

### 7. Technical observations

Python 3.14.7 will remain installed on the
system. The project will use an isolated
Python 3.11 Conda environment.

TensorFlow 2.15.1 and Keras 2.15.0 imported
successfully. The Hugging Face TensorFlow class
TFAutoModelForSequenceClassification was also
imported successfully.

PyArrow 26.0.1 initially caused an incompatibility
with NumPy 1.26.4. It was replaced with PyArrow
15.0.2, after which the project dependencies
imported successfully.

No training experiments have been performed
in this commit.

Hardware used for model training will be
documented when the experiments begin.


## IT2-10 - Training Utilities

**Date:** 2026-11-17

**Owner:** Daniel Varela

**GitHub user:** Danielvarelavictoria

### 1. Scope

This commit adds reusable training utilities in `src/train.py`:

- `set_seed(seed=1234)`: sets Python, NumPy and TensorFlow random seeds and enables TensorFlow deterministic operations when supported.
- `get_logger(model_name, log_dir="logs")`: creates a model-specific logger that writes to `training_logs_<model>.txt`.
- `Timer`: context manager for measuring elapsed time in seconds.
- `save_grid_results(results, model_name, output_dir="outputs/metrics")`: exports hyperparameter-grid results to CSV and returns the saved path.

### 2. Implementation decisions

- The project seed is defined as `SEED = 1234`.
- Logger names and output filenames are sanitized to avoid unsafe path characters.
- Repeated calls to `get_logger` avoid attaching duplicate file handlers for the same log file.
- `save_grid_results` accepts either a Pandas DataFrame or a list of dictionaries.
- Grid results are saved without the DataFrame index using the filename `<model>_grid_results.csv`.
- `Timer` is used with a `with` block and exposes `elapsed_seconds` and `elapsed`.

### 3. Validation results

Validation was performed in the `goemotions-it2` Conda environment on Windows PowerShell.

- Import check: passed. `set_seed`, `get_logger`, `Timer`, and `save_grid_results` imported successfully.
- Reproducibility check: passed. Python `random`, NumPy, and TensorFlow generated identical values after resetting the seed to `1234`; output confirmed `Reproducible: True`.
- Logger check: passed. `get_logger('test_model')` wrote an INFO message without errors.
- Timer check: passed. The context manager measured approximately 1.00 seconds for a one-second sleep.
- CSV export check: passed. `save_grid_results` created `test_model_grid_results.csv` with rows for units 32 / accuracy 0.85 and units 64 / accuracy 0.90.
- Temporary test outputs were created outside the repository. The logger test file in `logs` is scheduled for removal before commit.

TensorFlow emitted informational oneDNN/CPU messages and a deprecation warning during the seed test; the test itself passed and these messages did not indicate a failure.

## IT2-11 - Exploratory Data Analysis (EDA)

**Date:** 2026-11-18

**Owner:** Daniel Varela

**GitHub user:** Danielvarelavictoria

### 1. Scope

Created `notebooks/01_eda.ipynb` to inspect the original GoEmotions data before modeling. The notebook uses the shared loading and sentiment-mapping functions from `src/preprocessing.py`. It does not modify the raw files or remove records.

The analysis covers the combined `train.tsv`, `dev.tsv` and `test.tsv` files, their structure and descriptive summary, missing and empty values, repeated identifiers and normalized texts, emotion frequencies, sentiment-group counts, and the masked tokens `[NAME]` and `[RELIGION]`.

### 2. Observed results

- Total comments loaded: 54,263.
- Columns: `comment_id`, `text`, `emotion_ids`, `emotions`, `sentiment` (5 columns).
- Defined emotions: 28.
- Missing values: 0 across all columns.
- Empty or whitespace-only text values: 0.
- Empty comment identifiers: 0.
- Empty emotion-id lists: 0.
- Repeated comment identifiers: 0.
- Repeated rows by normalized text: 320. This is a descriptive count; no rows were removed in the notebook.
- Sentiment counts matched the reference values exactly: positive 19,020; negative 10,217; ambiguous 4,805; neutral 16,021; mixed 4,200 (54,263 total).
- The most frequent individual emotion shown in the output was `neutral`, with 17,772 occurrences. Emotion occurrences count labels, not unique comments.
- `[NAME]`: present in 7,558 comments, with 9,598 total occurrences.
- `[RELIGION]`: present in 121 comments, with 132 total occurrences.

### 3. Validation

The notebook was opened in Jupyter using the `Python (goemotions-it2)` kernel and marked as Trusted. The user executed the notebook and shared the resulting outputs for review. The displayed cells completed through the masked-token examples, with no error visible in the supplied screenshots.

### 4. Interpretation and limitations

The sentiment-group mapping reproduces all five reference counts. No missing or empty values and no repeated comment identifiers were observed. The normalized-text check found 320 repeated rows, but this EDA notebook does not remove them. The masked tokens are counted and inspected without modifying the original texts. Cleaning, balancing and splitting remain responsibilities of the shared preprocessing pipeline.


## IT2-12 - EDA visualizations

**Date:** 2026-11-18

**Owner:** Daniel Varela

**GitHub user:** Danielvarelavictoria

### 1. Scope

Extended `notebooks/01_eda.ipynb` with visualizations based on the combined original GoEmotions dataset. The new section includes: (1) comment counts and percentages by sentiment group, (2) the 15 most frequent individual emotions, (3) the distribution of comment length measured in words, and (4) the number of emotion labels assigned per comment. A final cell prints calculated indicators to support interpretation.

### 2. Implementation decisions

- Used Matplotlib for the plots; no additional visualization dependency was introduced.
- Reused the DataFrame and emotion/sentiment counts created in IT2-11.
- Kept the original data unchanged and did not remove or filter records for the visualizations.
- Added a short interpretation after each chart, distinguishing emotion-label occurrences from unique comments.
- Used calculated percentages and summary statistics rather than estimating values from chart appearance.

### 3. Validation

Pending execution of the updated notebook in the `goemotions-it2` Conda environment. The visualizations and their output tables should be checked by running all cells from top to bottom.

### 4. Interpretation and limitations

The plots are descriptive. The sentiment chart compares the five project groups; the emotion chart counts individual label assignments; the text-length histogram describes comment word counts; and the label-count chart shows how many emotion labels each comment contains. These observations can inform later preprocessing, but they do not by themselves determine model-specific truncation or filtering decisions.


## IT2-13 - EDA insights and preprocessing justification

**Date:** 2026-11-19

**Owner:** Daniel Varela

**GitHub user:** Danielvarelavictoria

### 1. Scope

Extended `notebooks/01_eda.ipynb` with data-quality conclusions, preprocessing rationale, and calculation of the sequence parameters required by later RNN/LSTM/Transformer work. Added `outputs/metrics/eda_summary.json` and read it back in the notebook to validate its contents.

### 2. Parameter calculation

- `max_len`: 13, the 95th percentile of word counts in `clean_text` from the 800-row training split.
- `max_words`: 2302, the number of unique tokens in `clean_text` from the training split only.
- Training rows: 800; test rows: 200.
- Empty cleaned training texts: 0.
- Vocabulary scope excludes the test split to avoid using test information when defining the training vocabulary.

### 3. Insights and preprocessing decisions

- The original-data EDA found no null values, empty texts, or repeated comment IDs. It identified 320 repeated rows by normalized text; this EDA check was descriptive and did not remove records.
- The five sentiment-group counts matched the project reference values.
- The downstream binary dataset is balanced at 500 positive and 500 negative examples; its split contains 800 training and 200 test examples.
- The shared `clean_text` implementation lowercases text, expands negative contractions, removes punctuation and masked `[NAME]` / `[RELIGION]` markers, and removes common stopwords.
- Negations such as `no`, `not`, and `never` are retained because removing them can reverse the meaning of a sentiment-bearing phrase.
- `max_len` uses the 95th percentile of training word lengths as a compromise between sequence coverage and computational cost. Some longer examples may be truncated.
- `max_words` is based only on unique cleaned tokens in the training split; the test vocabulary is not used.

### 4. Validation

The notebook includes assertions that compare the values read from `outputs/metrics/eda_summary.json` with the values calculated in the same run. Run the notebook top-to-bottom in `Python (goemotions-it2)` and verify that the JSON is created and read successfully.

### 5. Output

- `outputs/metrics/eda_summary.json`
- Updated `notebooks/01_eda.ipynb`
- Updated `logs/notes_varela.md`


## IT2-19 - Vanilla RNN classifier architecture

**Date:** 2026-11-20

**Owner:** Daniel Varela

**GitHub user:** Danielvarelavictoria

### 1. Scope

Implemented `build_rnn` in the `# --- RNN ---` section of `src/models.py`. The Dense, LSTM and Transformer sections were left unchanged.

### 2. Architecture and implementation decisions

The model follows the required sequence: integer-token Input, Embedding, SimpleRNN, Dense with ReLU, Dropout, and a one-unit Dense output with sigmoid activation. It is compiled for binary classification with `binary_crossentropy` and Adam.

Default dimensions use the shared EDA summary: `max_words=2302` and `max_len=13`. Initial tunable defaults are `embedding_dim=64`, `rnn_units=64`, `dense_units=32`, `dropout_rate=0.3`, and `learning_rate=0.001`. The dimensions and training hyperparameters are exposed as function arguments so the later manual grid search can reuse the builder.

The input expects integer sequences generated by `build_vectorizer`; the vectorizer must be adapted on training texts only, as specified by the preprocessing pipeline. No deprecated `input_length` argument is used in the Embedding layer.

### 3. Validation

Pending local validation in the `goemotions-it2` environment: inspect `model.summary()` and run a forward pass with a batch of integer sequences. Record the observed output shape before committing.

## IT2-20 - Manual Keras Grid Search

**Date:** 2026-11-20

**Owner:** Daniel Varela

**GitHub user:** Danielvarelavictoria

### 1. Scope

Extended `src/train.py` with two reusable functions for tuning and fitting Keras sequence models:

- `run_keras_grid(build_fn, param_grid, X_train, y_train, model_name, **fit_kwargs)`: evaluates hyperparameter configurations using a validation subset created from the training data.
- `fit_best_keras(build_fn, best_params, X_train, y_train, model_name, **fit_kwargs)`: fits the selected configuration using training data and a validation subset for early stopping.

The functions receive the model builder as an argument so the same training workflow can be used with the vanilla RNN and the LSTM architecture.

### 2. Grid-search configuration

The default `KERAS_PARAM_GRID` follows `MASTER_PROMPT.md` section 6.3:

- `units`: 32, 64, 128.
- `embedding_dim`: 50, 100.
- `dropout`: 0.2, 0.5.
- `batch_size`: 16, 32, 64.
- `optimizer`: RMSprop, Adam, both with learning rate 1e-3.
- Maximum training duration: 30 epochs, with early stopping.

The grid contains 72 configurations when all parameter values are combined. For the RNN builder, the shared grid names are translated to the corresponding arguments `rnn_units` and `dropout_rate`. `batch_size` and `optimizer` are handled by the training function rather than passed to the model builder.

### 3. Validation and selection decisions

- A stratified validation split is created from `X_train` and `y_train`; the test set is not passed to either function.
- Each candidate uses `EarlyStopping` with `monitor="val_loss"`, `min_delta=0.0001`, and `restore_best_weights=True`.
- The validation predictions use a 0.5 threshold.
- The primary selection metric is Cohen's kappa on the validation subset. If configurations have the same kappa, the one with the lower validation loss is ranked first.
- The results table records each configuration's parameters, validation loss, accuracy, precision, recall, F1, kappa, training time, parameter count, and number of epochs trained.
- Results are saved as `outputs/metrics/<model_name>_grid_results.csv`.
- Training progress is recorded with the existing model-specific logger.

### 4. Validation status

The implementation has been prepared, but the two-configuration smoke test has not yet been run. The following checks remain pending:

- Import `run_keras_grid`, `fit_best_keras`, and `KERAS_PARAM_GRID` from `src.train`.
- Run a two-configuration grid with a small synthetic sequence dataset and a small RNN configuration.
- Confirm that the result table contains both configurations and the validation metrics, elapsed time, and parameter count.
- Confirm that the CSV is created in the requested output directory and that the model-specific training log is written.
- Confirm that `fit_best_keras` returns a fitted model and its Keras History object.

The full 72-configuration grid and the final test-set evaluation are not part of this smoke test. The test set remains reserved for the final evaluation in the model notebook.

### 5. Output

- Updated `src/train.py`
- Updated `logs/notes_varela.md`
- Grid-results CSV will be generated during the smoke test under `outputs/metrics/`.


### Validation
- Import check: passed. `KERAS_PARAM_GRID`, `run_keras_grid` and `fit_best_keras` imported successfully.
- Grid size check: passed. The configured grid contains 72 combinations.
- Smoke test: passed with synthetic data (40 samples, sequence length 13), two configurations, Adam, batch size 8 and a maximum of two epochs.
- The grid search returned two result rows with validation loss, accuracy, precision, recall, F1, Cohen's kappa, training time, trainable parameter count and epochs trained.
- `fit_best_keras` successfully returned a trained model and a non-empty training history.
- The test used only synthetic data; the project test set was not accessed.
- TensorFlow emitted informational/deprecation warnings, but execution completed successfully.
