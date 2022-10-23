import pandas as pd
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # this and
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # ----- this just to hide some unimportant warnings
import tensorflow as tf


ds = pd.read_csv("data/train_set.csv")

N_MODELS = 10  # if your to-be-classified dataset is huge, keep this number low.
i = 0
while i < N_MODELS:
    ds1 = ds.sample(frac=0.8).reset_index(drop=True)  # an 80% sample of the dataset to ensure variety of opinion

    X = ds1[["danceability", "energy", "loudness", "key", "time_signature", "speechiness", "acousticness",
             "instrumentalness", "liveness", "valence", "tempo"]]  # input columns
    X = (X - X.min()) / (X.max() - X.min())  # min-max normalizing all input values to [0, 1]
    X = X.to_numpy()

    Y = ds1[["mood"]]  # labels column
    Y = Y.to_numpy().flatten()

    SPLIT = int(0.8 * len(X))  # 80-20 train-test split. see how the train set is 0.64x the whole dataset now.
    X_train, Y_train = X[:SPLIT], Y[:SPLIT]
    X_test, Y_test = X[SPLIT:], Y[SPLIT:]

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(11, )),
        tf.keras.layers.Dense(32, activation="LeakyReLU"),
        tf.keras.layers.Dense(5, activation="softmax"),
    ])

    model.compile(optimizer="Adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])

    model.fit(X_train, Y_train, epochs=200, batch_size=32, verbose=0)
    loss, acc = model.evaluate(X_test, Y_test, batch_size=1, verbose=0)

    if loss < 0.5 and acc > 0.8:  # filtering out high loss low accuracy models and saving only the good ones
        model.save(f"saved_models/{i}")
        print(i, loss, acc)
        i += 1
