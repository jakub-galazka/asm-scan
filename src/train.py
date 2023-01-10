import os
import sys
import time
import random
import numpy as np
import tensorflow as tf
from history import history
from util.json import to_json
from util.data_loader import load_data
from util.tokenizer import load_tokenizer
from util.preprocessing import pad_protein_sequences
from util.file_explorer import NEG_DATA_DIR, POS_DATA_DIR, MODELS_DIR, DATA_HIST_DIR, CONFIG_FILENAME, makedir

# Model params
MODEL_NAME = "bass-model"
CV_ITERS = 6
D = 8
M = 10
EPOCHS = 40

def train(model_dir: str, cv_iters: int) -> None:
    tokenizer = load_tokenizer()
    V = len(tokenizer.word_index) + 1

    model = None
    for cv_i in range(1, cv_iters + 1):
        # Load data
        x_trn, y_trn = load_data(
            os.path.join(NEG_DATA_DIR, "PB40", f"PB40_1z20_clu50_trn{cv_i}.fa"),
            os.path.join(POS_DATA_DIR, "bass_motif", f"bass_ctm_motif_trn{cv_i}.fa")
        )
        x_val, y_val = load_data(
            os.path.join(NEG_DATA_DIR, "PB40", f"PB40_1z20_clu50_val{cv_i}.fa"),
            os.path.join(POS_DATA_DIR, "bass_motif", f"bass_ctm_motif_val{cv_i}.fa")
        )

        # Pad protein sequences
        T = len(max(x_trn, key=len))
        x_trn = pad_protein_sequences(x_trn, T)
        x_val = pad_protein_sequences(x_val, T)

        # Tokenize text
        x_trn = np.asarray(tokenizer.texts_to_sequences(x_trn))
        x_val = np.asarray(tokenizer.texts_to_sequences(x_val))

        # Build model
        i = tf.keras.layers.Input(shape=(T,), name="input")
        x = tf.keras.layers.Embedding(V, D, name="embedding")(i)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(M, return_sequences=True), name="bi-lstm")(x)
        x = tf.keras.layers.LSTM(int(M / 2), name="lstm")(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid", name="classif")(x)
        model = tf.keras.models.Model(i, x, name=MODEL_NAME)

        model.compile(
            loss="binary_crossentropy",
            optimizer="adam",
            metrics=[
                tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
                tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
                tf.keras.metrics.SensitivityAtSpecificity(.99, name="sens_at_spec_99"),
                tf.keras.metrics.SensitivityAtSpecificity(.999, name="sens_at_spec_999")
            ]
        )

        # Train model
        print(f"\n_________________________ CV Iteration {cv_i} / {cv_iters} _________________________\n")
        r = model.fit(
            x_trn,
            y_trn,
            epochs=EPOCHS,
            validation_data=(x_val, y_val)
        )

        # Save results
        model_name = MODEL_NAME + str(cv_i)
        model.save(makedir(os.path.join(model_dir, MODELS_DIR, model_name)))
        np.save(makedir(os.path.join(model_dir, DATA_HIST_DIR, model_name)), r.history)

    # Save model info
    with open(os.path.join(model_dir, "architecture.txt"), "w") as f:
        model.summary(print_fn=lambda line: f.write(line + "\n"))

    to_json(os.path.join(model_dir, CONFIG_FILENAME), {"model_name": MODEL_NAME, "T": T})

def test_mode() -> bool:
    if ("--test" in sys.argv) or ("-t" in sys.argv):
        seed = 1
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        return True
    return False

if __name__ == "__main__":
    model_dir = os.path.join(MODELS_DIR, str(time.time()))

    cv_iters = CV_ITERS
    if test_mode():
        cv_iters = 1
        model_dir += "-test"

    train(model_dir, cv_iters)
    history(model_dir)
