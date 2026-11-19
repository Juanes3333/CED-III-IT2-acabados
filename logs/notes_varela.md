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
