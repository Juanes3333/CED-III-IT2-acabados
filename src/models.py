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


def build_rnn(
    max_words: int = 2302,
    max_len: int = 13,
    embedding_dim: int = 64,
    rnn_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
):
    """Build a compiled vanilla RNN classifier for binary sentiment.

    The model expects integer token sequences produced by the project's
    ``build_vectorizer`` function. Fit that vectorizer on training texts only
    before passing the resulting sequences to this model.

    Args:
        max_words: Maximum vocabulary size used by TextVectorization.
        max_len: Fixed sequence length produced by TextVectorization.
        embedding_dim: Size of each learned token embedding.
        rnn_units: Number of SimpleRNN units.
        dense_units: Number of units in the hidden Dense layer.
        dropout_rate: Dropout fraction applied before the output layer.
        learning_rate: Learning rate for the Adam optimizer.

    Returns:
        A compiled Keras Sequential model with a sigmoid binary output.

    Raises:
        ValueError: If a dimension is not positive or dropout_rate is outside
            the interval [0, 1).
    """
    from tensorflow import keras

    if max_words <= 0 or max_len <= 0:
        raise ValueError("max_words and max_len must be positive")
    if embedding_dim <= 0 or rnn_units <= 0 or dense_units <= 0:
        raise ValueError("embedding_dim, rnn_units and dense_units must be positive")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the interval [0, 1)")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(max_len,), dtype="int32", name="tokens"),
            keras.layers.Embedding(
                input_dim=max_words,
                output_dim=embedding_dim,
                name="embedding",
            ),
            keras.layers.SimpleRNN(rnn_units, name="simple_rnn"),
            keras.layers.Dense(dense_units, activation="relu", name="dense_hidden"),
            keras.layers.Dropout(dropout_rate, name="dropout"),
            keras.layers.Dense(1, activation="sigmoid", name="sentiment_output"),
        ],
        name="vanilla_rnn_classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --- LSTM ---



def build_lstm(
    max_words: int = 2302,
    max_len: int = 13,
    embedding_dim: int = 64,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
):
    """Build a compiled LSTM classifier for binary sentiment.

    The input consists of integer token sequences from the project's
    training-only-fitted text vectorizer. The LSTM replaces the SimpleRNN
    cell while keeping the rest of the RNN architecture consistent.
    """
    from tensorflow import keras

    if max_words <= 0 or max_len <= 0:
        raise ValueError("max_words and max_len must be positive")
    if embedding_dim <= 0 or lstm_units <= 0 or dense_units <= 0:
        raise ValueError("embedding_dim, lstm_units and dense_units must be positive")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in the interval [0, 1)")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(max_len,), dtype="int32", name="tokens"),
            keras.layers.Embedding(
                input_dim=max_words,
                output_dim=embedding_dim,
                name="embedding",
            ),
            keras.layers.LSTM(lstm_units, name="lstm"),
            keras.layers.Dense(dense_units, activation="relu", name="dense_hidden"),
            keras.layers.Dropout(dropout_rate, name="dropout"),
            keras.layers.Dense(1, activation="sigmoid", name="sentiment_output"),
        ],
        name="lstm_classifier",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --- Transformer ---
