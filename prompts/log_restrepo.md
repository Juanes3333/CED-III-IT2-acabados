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

### [2026-11-17] IT2-07: Cleaning, tokenization and split

- **Member:** Restrepo
- **Tool / Model:** Claude Code
- **Role of AI:** Technical Assistance & Code Review

#### Prompt
> "I'm working on the text preprocessing stage. Help me review the cleaning and tokenization criteria, especially how to preserve negations, and clarify the requirements for the stratified dataset split and vectorizer."

#### AI Contribution
Claude Code was used as a technical support tool to clarify some aspects of text preprocessing and data preparation. It provided suggestions regarding punctuation handling, stopword removal, tokenization and the importance of preserving negations during cleaning. It also helped review the stratified split configuration and the separation between training data and vectorizer adaptation.

#### Review & Critical Refinements
- **Corrected:** The stopword removal process was adjusted to preserve negations such as `not`, `never` and `no`, since removing them could change the original sentiment of a comment.
- **Adjusted:** Negative contractions are handled before punctuation removal so expressions such as `don't` do not lose their negation. Curly apostrophes are also normalized first.
- **Adjusted:** The dataset split uses stratification and a fixed seed to maintain the sentiment distribution and reproducibility. The vectorizer is adapted only using training data to avoid data leakage.
- **Decided:** The placeholders `[NAME]` and `[RELIGION]` are excluded from the processed text. The dataset contains 143 occurrences of `[NAME]`, and these placeholders do not contribute useful sentiment information.
- **Corrected:** During testing, one of the sanity checks was found to be incorrect because `again` is a stopword. Therefore, the expected result for `"Never again!"` should retain `never`. The test was corrected without changing the preprocessing logic.
- **Not validated:** The `build_vectorizer` function could not be fully tested because TensorFlow was not installed in the environment.

#### Discarded
The use of NLTK's stopword corpus was discarded because it requires an additional download and includes negations that would need to be manually preserved. An alternative stopword handling approach was preferred.