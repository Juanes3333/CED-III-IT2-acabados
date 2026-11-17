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
