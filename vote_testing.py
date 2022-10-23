"""in libraries such as xgb where constructs for random forests come inbuilt, this file would be one or two lines to
evaluate the accuracy of a forest. In fact this whole project would probably be a handful of lines. it is probably
possible in tensorflow as well, but I'm new to the subject and unaware of such a construct. Either way, doing it
manually gives you a good sense of what is actually going on when a "random forest" votes on the predicted label.

Since I have made the code itself very readable, here is the full explanation: The 10 trained models are one after
the other given each input from the dataset at a time, and each of them "votes" on which answer it believes is
correct. The winning vote is ultimately chosen as the forest's verdict, and vetted against the true answer. Wrong
verdicts are counted, giving us the forest's accuracy. """

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
# again, terrible practice
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
