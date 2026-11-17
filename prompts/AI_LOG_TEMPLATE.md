# AI Interaction Log Template

This template records Generative AI usage across the project (Course GenAI Level 3: Collaboration with GenAI). It focuses on AI assistance with **conceptual explanations, understanding requirements, debugging guidance, solution reviews, and drafting substantial code or documentation**.

Routine command execution and minor formatting edits do not need to be logged.

---

## Template (Copy & Paste)

```markdown
### [YYYY-MM-DD] IT2-NN: <Topic / Task>

- **Member:** <Restrepo / Varela / Granda>
- **Tool / Model:** <e.g., Antigravity CLI / Gemini 3.8 Flash, ChatGPT, Claude>
- **Role of AI:** <Explanation | Requirements Understanding | Debugging Guidance | Solution Review | Code/Doc Drafting>

#### Prompt
> <Prompt or question asked to the AI in our own words>

#### AI Contribution
<Concise summary of the explanation, advice, or generated code/documentation>

#### Review & Critical Refinements
- **Accepted:** <What was kept and why>
- **Adjusted:** <What was changed or corrected, including fixes to slide errors or project specifics>
- **Discarded:** <What was rejected and why>
```

---

## Reference Example

### [2026-11-17] IT2-03: README and Dataset Documentation Scaffolding

- **Member:** Granda
- **Tool / Model:** Antigravity CLI / Gemini 3.8 Flash
- **Role of AI:** Requirements Understanding & Documentation Drafting

#### Prompt
> "Create README.md and data/README.md following MASTER_PROMPT.md §2 and §6.2. Include team table, exact directory tree, environment setup, expected GoEmotions counts (19,020 / 10,217 / 4,805 / 16,021 / 4,200), and Demszky et al. (ACL 2020) citation. Use placeholders for preprocessing details."

#### AI Contribution
The model generated full initial drafts for `README.md` (project overview, directory structure, environment setup, execution steps) and `data/README.md` (Demszky citation, emotion taxonomy mapping, class distribution counts table, balanced subset schema).

#### Review & Critical Refinements
- **Accepted:** Complete repository tree matching the course specification, official Demszky citation, exact 5-class counts, and dual environment instructions (Conda and Pip).
- **Adjusted:** Removed markdown trailing spaces to eliminate git whitespace warnings (`git diff --check`). Standardized placeholders to `TODO(IT2-07)` and `TODO(IT2-32)`.
- **Discarded:** Generic placeholder text for model descriptions; replaced with explicit references to the four required architectures (Dense NN, Vanilla RNN, LSTM, DistilBERT).
