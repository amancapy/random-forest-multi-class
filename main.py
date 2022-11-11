import pandas as pd
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf


ds = pd.read_csv("data/train_set.csv")

N_MODELS = 10
for i in range(N_MODELS):
    ds1 = ds.sample(frac=0.8).reset_index(drop=True)

    X = ds1[["danceability", "energy", "loudness", "key", "time_signature", "speechiness", "acousticness",
             "instrumentalness", "liveness", "valence", "tempo"]]
    X = (X - X.min()) / (X.max() - X.min())  # min-max scaling features to [0, 1]
    X = X.to_numpy()

    Y = ds1[["mood"]]
    Y = Y.to_numpy()

    SPLIT = int(0.8 * len(X))  # 80-20 split
    X_train, Y_train = X[:SPLIT], Y[:SPLIT]
    X_test, Y_test = X[SPLIT:], Y[SPLIT:]

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(11, )),
        tf.keras.layers.Dense(32, activation="LeakyReLU"),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])

    model.compile(optimizer="Adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

    model.fit(X_train, Y_train, epochs=200, batch_size=32, verbose=0)
    loss, acc = model.evaluate(X_test, Y_test, batch_size=1, verbose=0)

    model.save(f"savedmodels/{i}")
