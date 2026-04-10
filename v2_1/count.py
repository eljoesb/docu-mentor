"""count.py — Lee un CSV de labels y muestra los conteos. Nada más."""

import csv, sys

counts = {"b": 0, "t": 0, "m": 0}
for row in csv.DictReader(open(sys.argv[1])):
    if row["label"] in counts:
        counts[row["label"]] += 1
total = sum(counts.values())
if total != 21:
    print(f"  ⚠  WARNING: {total} etiquetas, esperaba 21. CSV incompleto?")
print(f"bueno: {counts['b']}/{total}  tangencial: {counts['t']}/{total}  malo: {counts['m']}/{total}")
