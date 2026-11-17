# Experiment and Technical Notes Template

> **Purpose:** Record engineering decisions, hypotheses, experiment configurations, hardware setups, and validation results across all commits.<br>
> **Rule:** Every commit should have a brief corresponding entry explaining what technical choices were made and the empirical evidence supporting them.

---

## Blank Entry Template (Copy & Paste)

```markdown
## IT2-NN: <Title of Commit or Task>

- **Date:** YYYY-MM-DD
- **Owner:** <Restrepo / Varela / Granda>
- **Branch:** <branch-name>
- **Commit Reference:** Refs: IT2-NN

### 1. Technical Decision / Hypothesis
- <What design choice, architecture, hyperparameter, or implementation decision was made?>

### 2. Context & Rationale
- <Why this decision was chosen; reference slides, MASTER_PROMPT.md §3-§7, or rubric requirements.>

### 3. Implementation Details
- Files modified/created: `<list of files>`
- Key parameters / configurations: `<e.g., SEED=1234, hidden_units=64, learning_rate=1e-3>`

### 4. Verification & Empirical Evidence
- Command(s) executed: `<exact CLI commands run>`
- Output / metrics observed: `<paste key terminal output, test results, or metric values>`
- Acceptance criteria checklist:
  - [ ] Criterion 1
  - [ ] Criterion 2

### 5. Observations & Teammate Hand-off
- <Any notes on dependencies, merge points (M#), or things intentionally left for future commits.>
```

---

## Example Entry (Reference)

## IT2-02: Environment Configuration

- **Date:** 2026-11-17
- **Owner:** Daniel Varela
- **Branch:** `chore/environment`
- **Commit Reference:** Refs: IT2-02

### 1. Technical Decision / Hypothesis
- Isolate runtime to Python 3.11 using Conda (`environment.yml`) to support TensorFlow 2.15 and Keras 2.15 without wheel incompatibility issues present in local Python 3.14.

### 2. Context & Rationale
- Slide 18-20 workflows use Keras and TensorFlow. Python 3.14 does not provide prebuilt TensorFlow binaries for Windows/macOS. Python 3.11 ensures native compatibility across teammates' operating systems.

### 3. Implementation Details
- Files created: `environment.yml`, `requirements.txt`, `logs/notes_varela.md`
- Key packages pinned: `tensorflow==2.15.1`, `keras==2.15.0`, `transformers==4.38.2`, `pyarrow==15.0.2` (for NumPy 1.26.4 compatibility).

### 4. Verification & Empirical Evidence
- Commands executed:
  ```bash
  conda env create -f environment.yml
  pip install -r requirements.txt
  python -c "import tensorflow as tf, keras, transformers, nltk; print('All imports succeeded')"
  ```
- Acceptance criteria checklist:
  - [x] Python pinned to 3.11
  - [x] Clean imports of TensorFlow, Keras, and Transformers
  - [x] NLTK tokenizers (`punkt`, `stopwords`) verified
  - [x] No broken dependency conflicts (`pip check`)

### 5. Observations & Teammate Hand-off
- Merged via PR #1 into `main` (Merge **M1**). Ready for scaffolding and data pipeline branches.
