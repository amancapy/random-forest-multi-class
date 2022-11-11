import pandas as pd
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

ds = pd.read_csv("data/train_set.csv")

X = ds[["danceability", "energy", "loudness", "key", "time_signature", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]]
X = (X - X.min()) / (X.max() - X.min())
X = X.to_numpy()

Y = ds[["mood"]]
Y = Y.to_numpy().flatten()


N_MODELS = 10
models = [tf.keras.models.load_model(f"saved_models/{i}", compile=False) for i in range(N_MODELS)]
outs = [model.predict(X, batch_size=1, verbose=1) for model in models]

wrongs = [0 for _ in range(4)]
for i in range(len(X)):
    moods = [0 for _ in range(4)]
    for j in range(N_MODELS):
        for k in range(4):
            if outs[j][i][k] == max(outs[j][i]):
                moods[k] += 1
                break

    guess = 0
    for k in range(4):
        if moods[k] == max(moods):
            guess = k
            break

    if Y[i] != guess:
        wrongs[guess] += 1

print("forest's accuracy: %.2f" % ((1 - (sum(wrongs) / len(X))) * 100) + "%")
