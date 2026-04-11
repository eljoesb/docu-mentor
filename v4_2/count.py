"""count.py — Lee un CSV de labels y muestra los conteos. Detecta el eje por las etiquetas."""

import csv, sys

rows = list(csv.DictReader(open(sys.argv[1])))
labels = [r["label"] for r in rows if r.get("label")]
total = len(labels)

if total != 21:
    print(f"  WARNING: {total} etiquetas, esperaba 21. CSV incompleto?")

# Detectar eje por las etiquetas presentes
if any(l in ("b", "t", "m") for l in labels):
    counts = {"b": 0, "t": 0, "m": 0}
    names = {"b": "bueno", "t": "tangencial", "m": "malo"}
elif any(l in ("f", "x", "i") for l in labels):
    counts = {"f": 0, "x": 0, "i": 0}
    names = {"f": "fiel", "x": "mixto", "i": "inventado"}
elif any(l in ("u", "p", "d") for l in labels):
    counts = {"u": 0, "p": 0, "d": 0}
    names = {"u": "util", "p": "parcial", "d": "danino"}
else:
    print(f"Error: etiquetas no reconocidas: {set(labels)}")
    sys.exit(1)

for l in labels:
    if l in counts:
        counts[l] += 1

parts = [f"{names[k]}: {v}/{total}" for k, v in counts.items()]
print("  ".join(parts))
