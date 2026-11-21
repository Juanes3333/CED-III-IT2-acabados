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

### [2026-11-19] IT2-16: Dense neural network on TF-IDF features

- **Member:** Restrepo
- **Tool / Model:** Claude Code
- **Role of AI:** Technical Assistance & Code Review

#### Prompt
> "Can you help me review the configuration of a dense neural network using TF-IDF and MLPClassifier? I want to make sure the pipeline handles vectorization and training correctly, supports parameter configuration by name and avoids data leakage."

#### AI Contribution
Claude Code was used as a technical support tool to clarify some aspects of combining TF-IDF vectorization with a dense neural network. It provided suggestions regarding the use of `Pipeline`, parameter configuration and reproducibility. It also helped review the training process and some validation checks to ensure that the model could be configured and fitted correctly.

#### Review & Critical Refinements
- **Accepted:** The use of `TfidfVectorizer` and `MLPClassifier` within a single `Pipeline`, ensuring that the vocabulary is learned only from training data during fitting and cross-validation, preventing data leakage.
- **Adjusted:** `early_stopping=True` was retained because scikit-learn creates its validation subset from the training data. The test set is not used to determine when training stops.
- **Adjusted:** `random_state=SEED` was configured in the MLP to make the training results reproducible.
- **Adjusted:** Scikit-learn imports are placed inside the function so that `src.models` can be imported without requiring scikit-learn or TensorFlow immediately.
- **Validated:** The pipeline was tested with a toy sample, its parameters were checked by name, and a train-only fit was performed using the real training split. The test file was not accessed.
- **Not validated:** Testing was performed with scikit-learn 1.9.1 on Python 3.14 instead of the versions specified in `environment.yml` (1.4.2). This verification remains pending for IT2-17 and IT2-18.

#### Discarded
Nothing significant was discarded. Parameter tuning was left for IT2-17, keeping the current values as initial defaults for the model.

### [2026-11-19] IT2-17: Tune the dense network with GridSearchCV on kappa

- **Member:** Restrepo
- **Tool / Model:** Claude Code
- **Role of AI:** Technical Assistance & Code Review

#### Prompt
> "Can you explain how to configure GridSearchCV for the dense model? I need to use Cohen's kappa, keep the five folds stratified and make sure the search only uses the training data. Also, what is the best way to store the results and selected parameters?"

#### AI Contribution
Claude Code was used as a technical support tool to clarify some aspects of hyperparameter tuning and cross-validation. It provided suggestions regarding the configuration of `GridSearchCV`, the use of Cohen's kappa as the scoring metric and the organization of the experiment results. It also helped review the reproducibility settings and the process for saving the best parameters and cross-validation results.

#### Review & Critical Refinements
- **Adjusted:** The scoring configuration uses `make_scorer(cohen_kappa_score)` instead of passing `cohen_kappa_score` directly, since scikit-learn requires a scorer compatible with the estimator evaluation interface.
- **Adjusted:** `StratifiedKFold(shuffle=True, random_state=SEED)` was used to maintain class proportions across the five folds and ensure reproducibility.
- **Adjusted:** `return_train_score=True` was included to compare training and validation performance and identify possible overfitting.
- **Adjusted:** The saved JSON includes the best parameters along with relevant experiment information, such as the score, number of folds and seed. The parameter configuration was also checked for compatibility with `build_dense_pipeline`.
- **Accepted:** The parameter grid defined in `MASTER_PROMPT.md` §6, containing 48 candidate configurations.
- **Validated:** The experiment was tested with a toy sample and the real training split of 800 rows, without accessing the test set. A second single-threaded execution was also used to check the consistency of the results.
- **Interpretation:** 22 of the 48 configurations are within one standard deviation of the best result. Therefore, the selected configuration is not considered a clearly superior solution.
- **Not validated:** Final performance metrics on the test set, which are scheduled for IT2-18.

#### Discarded
Adding TF-IDF parameters such as `max_features` and `ngram_range` to the search grid was postponed because it would significantly increase the number of model evaluations and was outside the parameter grid defined in `MASTER_PROMPT.md`. This possibility can be reconsidered in future experiments if necessary.

### [2026-11-19] IT2-18: Dense network section in the model notebook

- **Member:** Restrepo
- **Tool / Model:** Claude Code
- **Role of AI:** Technical Assistance & Code Review

#### Prompt
> "What should I consider when documenting the Dense neural network results in the shared notebook? I need to include the evaluation metrics, confusion matrix, influential terms and the main conclusions, while keeping the other model sections available for the team."

#### AI Contribution
Claude Code was used as a technical support tool to review the notebook structure and clarify some aspects of presenting the Dense model results. It provided suggestions regarding the organization of the analysis, the evaluation metrics, the visualization of influential terms and the interpretation of the model's performance. It also helped review the reproducibility of the results and the organization of the shared notebook.

#### Review & Critical Refinements
- **Accepted:** The professor's step order, using question-style titles and separate cells for each stage, with the same evaluation metrics for training and test data.
- **Adjusted:** The tuning process reuses `run_dense_grid` with the previously established scorer, stratified folds and training-only data. The test set is evaluated only after the model configuration is fixed.
- **Adjusted:** The influential terms visualization was changed to a diverging log-odds chart because the initial probability-based approach produced saturated values that were not informative. The notebook also clarifies that this is a one-word probe, not a linear coefficient interpretation.
- **Adjusted:** The conclusions were revised to avoid claiming that regularization improved performance, since the differences between `alpha` values were within the observed variation.
- **Adjusted:** The explanation of empty cleaned texts was clarified. Since none were found in the dataset splits, `fillna` is described only as a safety measure.
- **Adjusted:** The seed configuration uses `random` and NumPy without importing TensorFlow, since the Dense section does not require it. The Keras sections must configure their own seed.
- **Accepted with limitations:** A rough normal-approximation interval for test accuracy (approximately ±0.06) was included as a reference for the small test set, without replacing the bootstrap intervals planned for IT2-24.
- **Validated:** The notebook was executed without errors and produced consistent results on a second run. The `dense_metrics.csv` file contains training and test results, test accuracy reached 0.720 compared with 0.515 for the best baseline, and the saved pipeline was reloaded with matching predictions.
- **Not validated:** The exact Conda environment specified in `environment.yml` and TensorFlow were not tested. Validation used a temporary Python 3.11 environment with the versions from `requirements.txt`.

#### Discarded
The MLP loss-curve figure and grid heatmap were discarded because they did not add relevant information for the evaluation criteria. Using the first-layer weights as direct TF-IDF word importance was also discarded because the hidden layer combines information from multiple words, making those weights unsuitable for direct per-word interpretation.

### [2026-11-21] IT2-25: DistilBERT sequence classifier builder

- **Member:** Restrepo
- **Tool / Model:** Claude Code
- **Role of AI:** Technical Assistance & Code Review

#### Prompt
> "What are the main considerations when building a DistilBERT sequence classifier with TensorFlow? I need to follow the professor's slide 23 configuration and make sure the tokenization and model functions work with our current project structure."

#### AI Contribution
Claude Code was used as a technical support tool to clarify some aspects of integrating DistilBERT into the project. It provided guidance on the tokenization process, model configuration and compatibility between TensorFlow and the Transformers library. It also helped review the implementation structure and identify some details to consider when testing the model.

#### Review & Critical Refinements
- **Accepted:** The main configuration from slide 23, including `truncation=True`, `padding="max_length"`, `num_labels=2`, a learning rate of `2e-5` and a loss function configured with `from_logits=True`.
- **Adjusted:** TensorFlow and `transformers` imports are placed inside the functions so that `src.models` can be imported without requiring these libraries immediately.
- **Adjusted:** `max_length` was made configurable, with a default value of 64 based on the EDA, instead of using a fixed value from the slides. A sparse loss was also selected because the labels are represented as integers (0 and 1).
- **Adjusted:** TensorFlow was retained as the framework after checking its compatibility with the pinned versions of the Transformers library, rather than switching to PyTorch without a specific reason.
- **Validated:** A forward pass using four comments produced logits with shape `(4, 2)`, and the model's `evaluate` method ran successfully with the compiled loss.
- **Not validated:** The exact Conda environment specified in `environment.yml`, as well as model fine-tuning and final performance metrics, which are scheduled for IT2-26.

#### Discarded
Returning both the tokenizer and the model from `build_transformer` was discarded in favor of keeping the functions separate. The tokenizer is handled by `tokenize_for_transformer`, maintaining a simpler interface consistent with `MASTER_PROMPT.md`.

### [2026-11-21] IT2-26: Transformer extension notebook

- **Member:** Restrepo
- **Tool / Model:** Claude Code
- **Role of AI:** Technical Assistance & Code Review

#### Prompt
> "What should I consider when extending the model notebook with the Transformer section? I need to follow the professor's structure, evaluate the fine-tuned model properly and make sure the discussion is supported by the actual results."

#### AI Contribution
Claude Code was used as a support tool to clarify some aspects of the Transformer experiment and notebook organization. It provided suggestions regarding tokenization, validation, model evaluation and result visualization. It also helped review the interpretation of attention matrices, the comparison with previous models and some technical issues encountered during notebook execution.

#### Review & Critical Refinements
- **Accepted:** The main configuration from slide 23, including `lr=2e-5`, `num_labels=2`, `from_logits=True`, `truncation=True` and `padding="max_length"`, together with a fixed seed.
- **Corrected:** The number of training epochs is selected using a stratified validation split taken exclusively from the training data. `restore_best_weights=True` is used, and the test set is evaluated only once after training.
- **Adjusted:** The `max_length` value was determined from the training token lengths, with a mean of 19 and a maximum of 46, instead of blindly copying a fixed value from the slides. This ensures that no comment is truncated.
- **Adjusted:** A grid search was not considered appropriate because the model has approximately 67 million parameters and the available training dataset contains only 640 rows. The standard configuration was retained as the selected configuration.
- **Adjusted:** The confidence interval for the 200-comment test set uses the project's `bootstrap_ci` function to maintain consistency with the evaluation of the other models.
- **Adjusted:** The attention analysis was revised after examining the actual figures. The matrices showed considerable attention to `[SEP]` and punctuation rather than emotional words, so the discussion avoids treating attention as a faithful explanation of the model's predictions.
- **Adjusted:** The parameter-to-comment ratio was corrected to approximately 84,000. Training time was also reported as a range of 5.7 to 6.8 minutes based on repeated executions.
- **Adjusted:** Library warnings containing local file paths were silenced, and the notebook was checked for personal paths in its outputs.
- **Rejected:** The claim that the Transformer performs better solely because of its architecture. Pretraining on external data is another relevant factor that this experiment cannot isolate.
- **Validated:** The notebook executed from beginning to end without errors. Five runs using the same seed produced consistent metrics. Test accuracy reached 0.895 and Cohen's kappa reached 0.790. The bootstrap confidence interval did not overlap with the LSTM interval.
- **Not validated:** Execution in the exact Conda environment specified in `environment.yml`, GPU execution and results using alternative random seeds.

#### Discarded
A multi-seed experiment and learning-rate grid were discarded due to computational cost and time constraints. Attention rollout and per-head visualizations were also excluded because they did not provide additional evidence relevant to the evaluation criteria. Saving the fine-tuned model weights was discarded because of their large size and the requirement to avoid committing them to the repository.