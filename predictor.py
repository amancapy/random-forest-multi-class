import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

N_MODELS = 10
models = [tf.keras.models.load_model(f"saved_models/{i}", compile=False) for i in range(N_MODELS)]


for chunkn in range(5):
    ds = pd.read_csv(f"chunks/nogenre_chunks/chunk{chunkn}.csv").sample(frac=0.0001).reset_index(drop=True)[["danceability", "energy", "loudness", "key", "time_signature", "speechiness", "acousticness", "instrumentalness", "liveness",
         "valence", "tempo", "release_date", "name", "explicit", "duration_ms", "id"]]

    inp_cols = ds[["danceability", "energy", "loudness", "key", "time_signature", "speechiness", "acousticness",
                   "instrumentalness", "liveness", "valence", "tempo"]]
    inp_cols = (inp_cols - inp_cols.min()) / (inp_cols.max() - inp_cols.min())
    inp_cols = inp_cols.to_numpy()

    fin_cols = ds[["name", "explicit", "release_date", "duration_ms", "id"]]
    fin_cols = fin_cols.to_numpy()

    outs = [model.predict(inp_cols, batch_size=1, verbose=1) for model in models]

    final_set = open(f"outs/nogenre_outs/final_set{chunkn}_test1234.csv", "w", encoding="utf-8")

    for i in range(len(inp_cols)):
        moods = [0 for _ in range(4)]
        for j in range(N_MODELS):
            for k in range(4):
                if outs[j][i][k] == max(outs[j][i]):
                    moods[k] += 1 / N_MODELS / 2
                moods[k] += outs[j][i][k] / N_MODELS / 2

        for k in range(4):
            moods[k] = ("%.2f" % moods[k])

        line = ""
        for thing in fin_cols[i]:
            line += str(thing).replace(",", "-").replace("\"", "").replace(";", " ") + ","
        line += str(moods).replace("[", "").replace("]", "").replace("\'", "").replace(" ", "")
        final_set.writelines(line[:-1] + "\n")

