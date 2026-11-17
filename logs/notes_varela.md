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
