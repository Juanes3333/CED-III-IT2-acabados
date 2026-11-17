# Experiment notes - Granda

Entries follow `logs/NOTES_TEMPLATE.md` (added in IT2-04).

---

## IT2-03: Documentation and Dataset Scaffolding

- **Date:** 2026-11-17
- **Owner:** Samuel Granda
- **Branch:** `docs/scaffold`
- **Commit Reference:** Refs: IT2-03

### 1. Technical Decision / Hypothesis
- Implement comprehensive project and dataset documentation scaffolds (`README.md`, `data/README.md`) upfront to establish directory schemas, experimental guidelines, and citations before data pipeline implementation begins.

### 2. Context & Rationale
- `MASTER_PROMPT.md` §2 specifies the required repository tree. `data/README.md` must document dataset origin (Demszky et al., ACL 2020), emotion mappings, and expected corpus counts (positive: 19,020, negative: 10,217, ambiguous: 4,805, neutral: 16,021, mixed: 4,200; total: 54,263) as an empirical sanity check.

### 3. Implementation Details
- Files modified/created: `README.md`, `data/README.md`, `logs/notes_granda.md`
- Key parameters: `SEED = 1234`, balanced subset target (500 positive + 500 negative), 80/20 train/test split.

### 4. Verification & Empirical Evidence
- Verification script executed with automated assertions for DOIs, citations, and counts.
- `git diff --check` passed cleanly.

### 5. Observations & Teammate Hand-off
- Branch `docs/scaffold` remains open for IT2-04.

---

## IT2-04: AI Usage and Experiment Log Templates

- **Date:** 2026-11-17
- **Owner:** Samuel Granda
- **Branch:** `docs/scaffold`
- **Commit Reference:** Refs: IT2-04

### 1. Technical Decision / Hypothesis
- Provide standardized markdown templates (`prompts/AI_LOG_TEMPLATE.md`, `logs/NOTES_TEMPLATE.md`) with concrete blank forms and reference examples to enable teammates to log entries in <2 minutes while strictly complying with GenAI Level 3 audit criteria.

### 2. Context & Rationale
- GenAI Level 3 evaluation requires documenting both process and product: prompts submitted, summary of AI output, and critical refinement (what was kept, modified, or rejected and why). The experiment notes template provides standard technical structure across all 33 commits.

### 3. Implementation Details
- Files created: `prompts/AI_LOG_TEMPLATE.md`, `logs/NOTES_TEMPLATE.md`
- Files updated: `prompts/log_granda.md`, `logs/notes_granda.md`

### 4. Verification & Empirical Evidence
- Verified template clarity and fill time (<2 minutes per entry).
- Checked markdown formatting and whitespace with `git diff --check`.
- Acceptance criteria checklist:
  - [x] Blank template ready for copy-paste
  - [x] Concrete real-world example included
  - [x] Critique section (kept, modified, rejected) explicitly enforced
  - [x] Spacing check: commit IT2-04 occurs >1 hour after IT2-03 (12:57:14 vs current time > 14:15)

### 5. Observations & Teammate Hand-off
- IT2-04 is the final commit of branch `docs/scaffold`.
- Next step after pushing IT2-04: Open PR for Merge **M2** (`docs/scaffold` -> `main`), assigned to **Daniel Varela (V)** for review.
