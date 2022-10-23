f = open("data/bigset.csv", encoding="utf-8").read().split("\n")

fs = []
chunk_size = 250000

for i in range(1, len(f), chunk_size):
    fs.append(f[i:i+chunk_size])

for j in range(len(fs)):
    fchunk = open(f"chunks/chunk{j}.csv", "w", encoding="utf-8")
    fchunk.writelines(f[0] + "\n")

    for line in fs[j]:
        fchunk.writelines(line + "\n")