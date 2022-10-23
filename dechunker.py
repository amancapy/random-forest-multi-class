f = open("outs/xyz.csv", "w", encoding="utf-8")

lines = []
for i in range(5):
    fchunk = open(f"outs/xyz{i}.csv", encoding="utf-8").read().split("\n")

    for line in fchunk:
        lines.append(line + "\n")

for line in lines:
    f.writelines(line)

