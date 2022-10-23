import random
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

client_id = ...
client_secret = ...
username = ...
red_uri = "https://127.0.0.1:8080/"
token = SpotifyOAuth(
    username=username,
    client_id=client_id,
    client_secret=client_secret,
    redirect_uri=red_uri
)

spotify = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret),
    auth_manager=token
)

# if you include too many playlists, you need to be selective with the tracks, or quality deteriorates very quickly.
sadlists = [...]
chllists = [...]
haplists = [...]
hyplists = [...]


def save_attrs(listname, filename, ext):
    # get uris of all songs in all the playlists in listname.
    tracks = []
    for playlist in listname:
        for track in spotify.playlist_items(playlist)["items"]:
            tracks.append(track["track"]["uri"])
    tracks = list(set(tracks))

    # now ask spotify for each track's attributes
    txt = open(f"tracktributes/{filename}.txt", "w")
    count = 0
    for track in tracks:
        attrs = spotify.audio_features(track)[0]
        attrs = [attrs[key] for key in ["danceability", "energy", "loudness", "key", "mode", "time_signature", "speechiness", "acousticness",
                                        "instrumentalness", "liveness", "valence", "tempo", "duration_ms"]]
        line = ""
        for attr in attrs:
            line += str(attr) + ","
        line += f"{ext}\n"
        txt.writelines(line)
        count += 1
        print(f"{ext}: {count}/{len(tracks)}")

# get all of them at once
# for listname, filename, ext in zip(
#         [sadlists, chllists, haplists, hyplists],
#         ["sads", "shrs", "haps", "pass", "enrs"],
#         [0, 1, 2, 3]):
#     save_attrs(listname, filename, ext)


# once you have all of them in the same file, you need to filter out duplicates that may fall in different categories,
# which is terrible for dataset quality.
def drop_dupes():
    f = open("tracktributes_all.csv").read().split("\n")
    random.shuffle(f)

    fnew = []
    dupes = []
    for i in range(len(f)):
        for j in range(len(f)):
            if not i == j and f[i][:-1] == f[j][:-1]:
                dupes.append(f[i])
    print(len(dupes))
    for fline in f:
        if fline not in dupes:
            fnew.append(fline)
    f1 = open("train_set.csv", "w")
    f1.writelines("danceability,energy,loudness,key,mode,time_signature,speechiness,acousticness,instrumentalness,"
                  "liveness,valence,tempo,duration_ms,mood\n")
    for line in fnew:
        f1.writelines(line)
        f1.writelines("\n")

# drop_dupes()
