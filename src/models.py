"""Model architectures: Dense NN, vanilla RNN, LSTM and Transformer.

Each model lives in its own section below. Only the owner of the current
branch edits their section, so merges stay clean.
"""


# --- Dense ---

SEED = 1234


def build_dense_pipeline(**params) -> "sklearn.pipeline.Pipeline":
    """Build a TF-IDF + dense neural network (MLP) pipeline.

    The vectorizer and the network live in one ``Pipeline``. When the pipeline
    is fitted (or cross-validated) the TF-IDF vocabulary is learned only from
    the training rows of each fold, so nothing from the validation or test
    texts leaks into the features.

    Steps are named ``tfidf`` and ``mlp``. Any parameter can be set by name
    with the sklearn ``step__param`` syntax, for example
    ``build_dense_pipeline(mlp__hidden_layer_sizes=(32,), tfidf__max_features=2000)``.
    The same names work in a ``GridSearchCV`` ``param_grid``.

    Args:
        **params: Pipeline parameters in ``step__param`` form. They override
            the defaults below.

    Returns:
        An unfitted ``sklearn.pipeline.Pipeline``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    alpha=1e-3,
                    max_iter=300,
                    early_stopping=True,
                    n_iter_no_change=10,
                    random_state=SEED,
                ),
            ),
        ]
    )
    if params:
        pipeline.set_params(**params)
    return pipeline



# --- RNN ---



# --- LSTM ---



# --- Transformer ---
