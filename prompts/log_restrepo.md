# AI interaction log - Restrepo

Entries follow `prompts/AI_LOG_TEMPLATE.md` (added in IT2-04).

### [2026-11-17] IT2-05: Load GoEmotions and map emotions to sentiment

- **Member:** Restrepo
- **Tool / Model:** Claude Code
- **Role of AI:** Technical Assistance & Code Review

#### Prompt
> "I'm working on the GoEmotions data pipeline. Help me clarify the dataset structure and review some details about the sentiment mapping, especially the classification rules for emotions belonging to different groups."

#### AI Contribution
Claude Code was used as a support tool during development to clarify questions about the GoEmotions dataset format and some aspects of sentiment classification. It provided suggestions regarding file reading configurations, emotion ID handling and the mapping between emotions and sentiment groups. It also helped review specific implementation details and understand the expected classification behavior.

#### Review & Critical Refinements
- **Accepted:** The group assignment rule, where a sentiment group is assigned only when all emotions belong to the same group; otherwise, the result is classified as `mixed`.
- **Adjusted:** The file reading configuration after inspecting the actual dataset format and identifying that the files had no headers and IDs were stored as comma-separated text.
- **Adjusted:** The use of `quoting=3` and `dtype=str` to preserve the original text and avoid unwanted transformations during loading.
- **Added:** The `neutral` category manually because it was not included in the original JSON mapping.
- **Validated:** The five required counts by executing the implementation and comparing the results with the expected values.

#### Discarded
Nothing significant was discarded. The suggestions provided by the AI were reviewed and adapted to the actual dataset structure and project requirements.

### [2026-11-17] IT2-06: Balanced 500/500 sentiment dataset

- **Member:** Restrepo
- **Tool / Model:** Claude Code
- **Role of AI:** Technical Assistance & Code Review

#### Prompt
> "I'm working on building a balanced sentiment dataset. Help me review the deduplication criteria, the text normalization process and the sampling requirements to ensure the final dataset contains 500 positive and 500 negative comments."

#### AI Contribution
Claude Code was used as a support tool to clarify some aspects of the dataset preparation process, particularly text normalization, duplicate detection and balanced sampling. It provided suggestions for handling comments with conflicting sentiment labels and helped review the consistency of the CSV export process.

#### Review & Critical Refinements
- **Accepted:** Deduplication based on normalized text, following the project specification.
- **Adjusted:** Comments appearing with both positive and negative labels are excluded completely to avoid retaining potentially conflicting sentiment classifications.
- **Adjusted:** Rows are sorted by `comment_id` before sampling to ensure consistent results regardless of the original file order.
- **Adjusted:** The CSV export uses LF line endings to maintain consistency across different operating systems.
- **Validated:** The final dataset contains 500 positive and 500 negative comments, with the sampling process using the required seed.

#### Discarded
Nothing significant was discarded. The proposed approaches were reviewed and adapted according to the dataset requirements.