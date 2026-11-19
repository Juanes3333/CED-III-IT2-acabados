# AI Interaction Log — Samuel Granda

Entries follow `prompts/AI_LOG_TEMPLATE.md`. This log documents Generative AI assistance with explanations, requirements understanding, debugging guidance, solution reviews, and code/documentation drafting.

---

### [2026-11-17] IT2-03: README and Dataset Documentation Scaffolding

- **Member:** Samuel Granda
- **Tool / Model:** Antigravity CLI / Gemini 3.8 Flash
- **Role of AI:** Requirements Understanding & Documentation Drafting

#### Prompt
> Implement ONLY the files and changes explicitly assigned to IT2-03:
> - Read the exact IT2-03 requirements in .claude/COMMITS.md and follow all rules in .claude/CLAUDE.md and .claude/MASTER_PROMPT.md.
> - Create docs/scaffold from updated main.
> - Files: README.md, data/README.md.
> - Include project overview, team table, repo structure, environment setup, execution steps, expected GoEmotions counts, Demszky et al. (ACL 2020) citation, and preprocessing placeholders.

#### AI Contribution
The AI generated full initial drafts for `README.md` and `data/README.md`, including directory layout, citations, setup instructions, class distribution tables, and dataset schemas.

#### Review & Critical Refinements
- **Accepted:** Complete repository tree matching the specification, the Demszky et al. citation, the 5-class distribution counts, and Conda/Pip environment instructions.
- **Adjusted:** Fixed markdown line breaks to eliminate git whitespace warnings (`git diff --check`), and standardized placeholders to `TODO(IT2-07)` and `TODO(IT2-32)`.
- **Discarded:** Generic placeholder text for model architectures, replacing it with the specific project models (Dense NN, Vanilla RNN, LSTM, DistilBERT).

---

### [2026-11-17] IT2-04: Templates for AI Logging and Experiment Notes

- **Member:** Samuel Granda
- **Tool / Model:** Antigravity CLI / Gemini 3.8 Flash
- **Role of AI:** Requirements Understanding & Template Drafting

#### Prompt
> Implement ONLY IT2-04:
> - prompts/AI_LOG_TEMPLATE.md
> - logs/NOTES_TEMPLATE.md
> - My personal logs, only if required.
> Follow the exact requirements and acceptance criteria in COMMITS.md. Templates must be quick to fill (<2 minutes) and preserve mandatory Level 3 AI disclosure fields.

#### AI Contribution
The AI generated initial drafts for `prompts/AI_LOG_TEMPLATE.md` and `logs/NOTES_TEMPLATE.md`, providing boilerplate structures for prompt logging, AI output summaries, critical reviews, and technical experiment decisions.

#### Review & Critical Refinements
- **Accepted:** The core review fields (Prompt, AI Contribution, Review & Critical Refinements) and experiment note fields (Decision, Rationale, Verification).
- **Adjusted:** Simplified the language across the template and existing log entries to focus on explanations, requirements, debugging, and solution review rather than routine command logging.
- **Discarded:** Complex survey questionnaires and redundant checklist items that added unnecessary friction.

### [2026-11-18] IT2-08: Classification Metrics

- **Member:** Samuel Granda
- **Tool / Model:** Antigravity CLI / Claude Sonnet 4.6, ChatGPT
- **Role of AI:** Implementation Support & Code Review

#### Prompt
> Help with IT2-08 classification metrics, including Cohen's kappa, train/test evaluation, and CSV export, following the project requirements.

#### AI Contribution
Antigravity helped prepare the initial implementation of the evaluation functions. ChatGPT provided guidance for reviewing the code and checking the results.

#### Review & Critical Refinements
- **Accepted:** The evaluation functions and metric calculations.
- **Verified:** The expected results using different prediction examples.
- **Kept separate:** Visualizations and model comparison functionality for later tasks.

### [2026-11-18] IT2-09: Visualization Helpers

- **Member:** Samuel Granda
- **Tool / Model:** ChatGPT
- **Role of AI:** Development Support & Testing Guidance

#### Prompt
> Help me understand and implement the visualization requirements for IT2-09, following the project specifications.

#### AI Contribution
ChatGPT provided an initial code proposal and guidance for implementing and testing the visualization functions.

#### Review & Critical Refinements
- **Reviewed:** The seven functions against the project requirements.
- **Verified:** All seven functions were tested with synthetic data and generated PNG files successfully.
- **Decision:** Kept the visualization functionality separate from dataset analysis and model training, as required by the project.

---

### [2026-11-19] IT2-14: DummyClassifier Baseline Notebook

- **Member:** Samuel Granda
- **Tool / Model:** ChatGPT (GPT-6)
- **Role of AI:** Requirements Review, Notebook Drafting & Execution Guidance

#### Prompt
> Help prepare IT2-14 according to `.claude/COMMITS.md`. Create the complete `notebooks/02_baseline_model.ipynb` using `DummyClassifier` with `most_frequent`, `stratified`, and `uniform`. Follow build → fit → predict train → predict test, and report train/test metrics. Keep CSV export and confusion matrix for IT2-15. Provide complete files and paths so I can validate them together later.

#### AI Contribution
ChatGPT reviewed the existing evaluation and preprocessing interfaces and supplied an initial notebook file with the baseline experiment structure and explanatory Markdown. It also helped organize the IT2-14 documentation and proposed a later terminal-based validation workflow.

#### Review & Critical Refinements
- **Accepted for initial review:** Reusing the project's existing split, seed, and `evaluate_train_test` function instead of duplicating evaluation code.
- **Scope decision:** Kept metric persistence and the confusion matrix outside IT2-14 because those outputs belong to IT2-15.
- **Pending verification:** Run the notebook against the real processed data, inspect the observed metrics, and record any fixes or changes before committing. No experimental success is claimed yet.
