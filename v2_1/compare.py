"""
compare.py — Compara labels de classify.py contra predicciones.

Uso:
    python v2.1/compare.py v2.1/labels_bge_base_results.csv --predict 6
    python v2.1/compare.py labels_a.csv:3 labels_b.csv:1 labels_c.csv:2
"""

import argparse
import csv
import sys
from pathlib import Path

NAMES = {"b": "bueno", "t": "tangencial", "m": "malo"}


def load(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("label", "").strip() in ("b", "t", "m"):
                rows.append(row)
    return rows


def counts(rows):
    c = {"b": 0, "t": 0, "m": 0}
    for r in rows:
        c[r["label"]] += 1
    return c


def main():
    parser = argparse.ArgumentParser(description="Compara labels vs predicciones")
    parser.add_argument("files", nargs="+", help="archivo.csv o archivo.csv:prediccion")
    parser.add_argument("--predict", type=int, default=None,
                        help="Predicción de buenos (aplica al primer archivo)")
    args = parser.parse_args()

    systems = []
    for spec in args.files:
        if ":" in spec and spec.rsplit(":", 1)[1].isdigit():
            p, n = spec.rsplit(":", 1)
            systems.append((p, int(n)))
        else:
            systems.append((spec, None))

    if args.predict is not None and systems:
        systems[0] = (systems[0][0], args.predict)

    results = []
    for path_str, pred in systems:
        path = Path(path_str)
        if not path.exists():
            print(f"Error: {path} no existe", file=sys.stderr)
            continue
        rows = load(path)
        if not rows:
            print(f"Error: {path} sin labels válidas", file=sys.stderr)
            continue
        results.append((path.stem.replace("labels_", ""), counts(rows), pred, rows))

    print()
    # Tabla de conteos
    total_label = "total"
    print(f"  {'Sistema':<25} {'bueno':>7} {'tang':>7} {'malo':>7}")
    print(f"  {'─' * 48}")
    for name, c, _, _ in results:
        t = sum(c.values())
        print(f"  {name:<25} {c['b']:>4}/{t}  {c['t']:>4}/{t}  {c['m']:>4}/{t}")
    print()

    # Comparación con predicciones
    has_pred = any(p is not None for _, _, p, _ in results)
    if not has_pred:
        return

    print(f"  {'Sistema':<25} {'pred':>6} {'real':>6} {'Δ':>5}")
    print(f"  {'─' * 44}")
    for name, c, pred, rows in results:
        if pred is None:
            continue
        actual = c["b"]
        delta = actual - pred
        flag = " ✓" if abs(delta) <= 2 else " ← ERROR (Δ > 2)"
        print(f"  {name:<25} {pred:>5}  {actual:>5}  {delta:>+3}{flag}")

        if abs(delta) > 2:
            print()
            print(f"  *** Predicción erró por {abs(delta)}. Buenos encontrados: ***")
            for r in rows:
                if r["label"] == "b":
                    print(f"    Q{r['query_id']}: {r['query']}")
                    print(f"      source: {r['source']}")
            print()
    print()


if __name__ == "__main__":
    main()
