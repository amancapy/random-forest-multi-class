"""Since the trees in the forest are trained on different-enough chunks of the dataset, they will have slightly
different opinions on data. So you can get a sense not only of what class a datapoint falls into, but also of with
how much variety of opinion the forest makes this decision. For example, if 7 out of 10 trees agree that a song is
sad, I infer this as saying that this song is 70% sad and 30% whatever else. Although this may not be standard
practice, for my use case it's a convenient metric.

I've also used the sum of the residual values that trickle down to all the output nodes instead of just the "winning
node" since this gives me closer values such as 56% sad instead of just 70% or 80%. Since the loss metric being used
in the model is sparse cross-entropy, all output nodes' values "matter." If an output is [0.2 0.2 0.35 0.25] just
picking the argmax category as 1 vote is throwing away useful information, since the model found the input ambiguous.
This is useful since our label classes are not exclusive -- a song can be chill and happy.

Since random forests give only a small accuracy edge on individual models, the pain of excess computation is
compensated for by this collateral information even if it's not the most precise (as far as I know). """

import pandas as pd
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

N_MODELS = 10
models = [tf.keras.models.load_model(f"saved_models/{i}", compile=False) for i in range(N_MODELS)]
# loading models in a for loop is awful practice, keep that in mind. this is for simplicity.

# since my unlabelled dataset was large, I had to break it down into pieces and stitch it back together after labelling.
for chunkn in range(5):
    ds = pd.read_csv(f"chunks/...{chunkn}.csv").sample(frac=1).reset_index(drop=True)[["danceability", "energy", "loudness", "key", "time_signature", "speechiness", "acousticness", "instrumentalness", "liveness",
         "valence", "tempo", "release_date", "name", "explicit", "duration_ms", "id"]]

    inp_cols = ds[["danceability", "energy", "loudness", "key", "time_signature", "speechiness", "acousticness",
                   "instrumentalness", "liveness", "valence", "tempo"]]
    inp_cols = (inp_cols - inp_cols.min()) / (inp_cols.max() - inp_cols.min())  # min-max normalization again
    inp_cols = inp_cols.to_numpy()

    fin_cols = ds[["name", "explicit", "release_date", "duration_ms", "id"]]
    fin_cols = fin_cols.to_numpy()  # labels

    outs = [model.predict(inp_cols, batch_size=1, verbose=1) for model in models]
    # again, not the best practice

    final_set = open(f"outs/nogenre_outs/final_set{chunkn}.csv", "w", encoding="utf-8")

    for i in range(len(inp_cols)):
        moods = [0 for _ in range(4)]
        for j in range(N_MODELS):
            for k in range(4):
                if outs[j][i][k] == max(outs[j][i]):
                    moods[k] += 1 / N_MODELS / 2  # picking the argmax
                moods[k] += outs[j][i][k] / N_MODELS / 2  # picking up residuals

        for k in range(4):
            moods[k] = ("%.2f" % moods[k])  # down to two decimal points for saving in file

        # dirty string work for saving as file. this only has to happen once in the whole process so I can get away
        # with bad code here.
        line = ""
        for thing in fin_cols[i]:
            line += str(thing).replace(",", "-").replace("\"", "").replace(";", " ") + ","
        line += str(moods).replace("[", "").replace("]", "").replace("\'", "").replace(" ", "")
        final_set.writelines(line[:-1] + "\n")

