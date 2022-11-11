f = open("outs/nogenre_outs/fin_nogenre.csv", "w", encoding="utf-8")

lines = []
for i in range(5):
    fchunk = open(f"outs/nogenre_outs/final_set{i}.csv", encoding="utf-8").read().split("\n")

    for line in fchunk:
        lines.append(line + "\n")

for line in lines:
    f.writelines(line)

