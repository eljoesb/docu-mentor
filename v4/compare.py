"""
compare.py — Compara clasificaciones humanas (v3b) vs juez LLM (v4).

Lee tres pares de CSVs (humano vs juez, por eje) y produce:
1. Acuerdo por eje (% de coincidencia)
2. Acuerdo global (% sobre las 63 clasificaciones)
3. Matriz de confusion por eje
4. Lista de desacuerdos especificos

Uso:
    python v4/compare.py
"""

import csv
import sys
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).parent
V3_DIR = SCRIPT_DIR.parent / "v3"

AXES = {
    "retrieval": {
        "human": V3_DIR / "labels_retrieval_v3b.csv",
        "judge": SCRIPT_DIR / "judge_retrieval.csv",
        "labels": ["b", "t", "m"],
        "names": {"b": "bueno", "t": "tangencial", "m": "malo"},
    },
    "generation": {
        "human": V3_DIR / "labels_generation_v3b.csv",
        "judge": SCRIPT_DIR / "judge_generation.csv",
        "labels": ["f", "x", "i"],
        "names": {"f": "fiel", "x": "mixto", "i": "inventado"},
    },
    "utility": {
        "human": V3_DIR / "labels_utility_v3b.csv",
        "judge": SCRIPT_DIR / "judge_utility.csv",
        "labels": ["u", "p", "d"],
        "names": {"u": "util", "p": "parcial", "d": "danino"},
    },
}


def load_labels(path):
    labels = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            qid = int(row["query_id"])
            labels[qid] = row["label"].strip()
    return labels


def confusion_matrix(human_labels, judge_labels, valid_labels, names):
    """Imprime matriz de confusion."""
    matrix = Counter()
    for qid in human_labels:
        h = human_labels[qid]
        j = judge_labels.get(qid, "?")
        matrix[(h, j)] += 1

    # Header
    header = "         " + "  ".join(f"{names.get(l, l):>10}" for l in valid_labels)
    print(f"  {'':>10} {'Judge -->':>10}")
    print(f"  {'Human':>10} " + "  ".join(f"{names.get(l, l):>10}" for l in valid_labels))
    print(f"  {'':>10} " + "  ".join("-" * 10 for _ in valid_labels))

    for h_label in valid_labels:
        row = []
        for j_label in valid_labels:
            count = matrix.get((h_label, j_label), 0)
            row.append(f"{count:>10}")
        print(f"  {names.get(h_label, h_label):>10} " + "  ".join(row))


def main():
    total_agree = 0
    total_count = 0
    all_disagreements = []

    for axis, config in AXES.items():
        human_path = config["human"]
        judge_path = config["judge"]

        if not human_path.exists():
            print(f"Error: {human_path} no existe", file=sys.stderr)
            sys.exit(1)
        if not judge_path.exists():
            print(f"Error: {judge_path} no existe", file=sys.stderr)
            sys.exit(1)

        human = load_labels(human_path)
        judge = load_labels(judge_path)

        agree = 0
        disagree = []
        for qid in sorted(human.keys()):
            h = human[qid]
            j = judge.get(qid, "?")
            if h == j:
                agree += 1
            else:
                disagree.append((qid, h, j))

        count = len(human)
        pct = agree / count * 100 if count > 0 else 0

        print(f"\n{'=' * 55}")
        print(f"  {axis.upper()}: {agree}/{count} acuerdo ({pct:.0f}%)")
        print(f"{'=' * 55}")

        # Matriz de confusion
        print()
        confusion_matrix(human, judge, config["labels"], config["names"])

        # Desacuerdos
        if disagree:
            print(f"\n  Desacuerdos ({len(disagree)}):")
            for qid, h, j in disagree:
                h_name = config["names"].get(h, h)
                j_name = config["names"].get(j, j)
                print(f"    Q{qid:>2}: humano={h_name:<12} juez={j_name}")
                all_disagreements.append({
                    "axis": axis,
                    "query_id": qid,
                    "human": h,
                    "judge": j,
                })

        total_agree += agree
        total_count += count

    # Resumen global
    global_pct = total_agree / total_count * 100 if total_count > 0 else 0
    print(f"\n{'=' * 55}")
    print(f"  GLOBAL: {total_agree}/{total_count} acuerdo ({global_pct:.0f}%)")
    print(f"{'=' * 55}")

    # Tabla resumen
    print(f"\n  Resumen por eje:")
    print(f"  {'Eje':<15} {'Acuerdo':>10} {'%':>6}")
    print(f"  {'-'*15} {'-'*10} {'-'*6}")
    for axis, config in AXES.items():
        human = load_labels(config["human"])
        judge = load_labels(config["judge"])
        agree = sum(1 for qid in human if human[qid] == judge.get(qid))
        count = len(human)
        pct = agree / count * 100
        print(f"  {axis:<15} {agree:>5}/{count:<4} {pct:>5.0f}%")
    print(f"  {'TOTAL':<15} {total_agree:>5}/{total_count:<4} {global_pct:>5.0f}%")

    # Desacuerdos totales
    if all_disagreements:
        print(f"\n  Total desacuerdos: {len(all_disagreements)}/63")
    else:
        print(f"\n  Acuerdo perfecto en las 63 clasificaciones.")


if __name__ == "__main__":
    main()
