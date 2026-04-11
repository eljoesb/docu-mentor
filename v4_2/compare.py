"""
compare.py — Compara retrieval hybrid (v4.2) vs BGE-only (v3b).

Muestra:
1. Tabla query-por-query: label v3b vs v4.2, si el chunk cambio, tipo de fusion
2. Conteos por label
3. Gains y losses (buenos ganados y perdidos)
4. Comparacion con prediccion

Uso:
    python v4.2/compare.py
    python v4.2/compare.py --predict 7
"""

import argparse
import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
V3B_LABELS = SCRIPT_DIR.parent / "v3" / "labels_retrieval_v3b.csv"
HYBRID_LABELS = SCRIPT_DIR / "labels_retrieval.csv"
HYBRID_RESULTS = SCRIPT_DIR / "hybrid_results.jsonl"


def load_labels(path):
    labels = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            qid = int(row["query_id"])
            labels[qid] = {
                "label": row["label"].strip(),
                "source": row.get("source", ""),
                "query": row.get("query", ""),
            }
    return labels


def load_hybrid_results(path):
    results = {}
    if not path.exists():
        return results
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                results[r["id"]] = r
    return results


def short_source(s):
    if not s:
        return "?"
    for marker in ("docs-source/",):
        idx = s.find(marker)
        if idx != -1:
            return s[idx + len(marker):]
    return Path(s).name


def label_name(l):
    return {"b": "bueno", "t": "tang", "m": "malo"}.get(l, l)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predict", type=int, default=7,
                        help="Prediccion de buenos (default: 7)")
    args = parser.parse_args()

    v3b = load_labels(V3B_LABELS)
    v42 = load_labels(HYBRID_LABELS)
    hybrid = load_hybrid_results(HYBRID_RESULTS)

    if not v3b or not v42:
        print("Error: faltan archivos de labels")
        return

    qids = sorted(set(v3b.keys()) | set(v42.keys()))

    # --- Tabla query-por-query ---
    print()
    print(f"  {'Q':>3} {'v3b':>6} {'v4.2':>6} {'chunk':>8} {'tipo':>12}  {'query':<50}")
    print(f"  {'─' * 90}")

    gains = []
    losses = []
    upgrades = []
    downgrades = []

    for qid in qids:
        l3 = v3b.get(qid, {}).get("label", "?")
        l4 = v42.get(qid, {}).get("label", "?")
        query = v3b.get(qid, {}).get("query", v42.get(qid, {}).get("query", ""))

        hr = hybrid.get(qid, {})
        chunk_changed = "same"
        change_type = hr.get("change_type", "")
        if hr:
            src_hyb = short_source(hr.get("chunk_source", ""))
            src_den = short_source(hr.get("dense_source", ""))
            if src_hyb != src_den:
                chunk_changed = "CHANGED"

        # Classify the transition
        marker = ""
        if l3 == "b" and l4 != "b":
            marker = " << LOSS"
            losses.append(qid)
        elif l3 != "b" and l4 == "b":
            marker = " << GAIN"
            gains.append(qid)
        elif l3 != l4:
            if l4 == "m" and l3 == "t":
                downgrades.append(qid)
            elif l4 == "t" and l3 == "m":
                upgrades.append(qid)

        print(f"  {qid:>3} {label_name(l3):>6} {label_name(l4):>6} "
              f"{chunk_changed:>8} {change_type:>12}  {query:<50}{marker}")

    # --- Conteos ---
    c3 = {"b": 0, "t": 0, "m": 0}
    c4 = {"b": 0, "t": 0, "m": 0}
    for qid in qids:
        l3 = v3b.get(qid, {}).get("label", "")
        l4 = v42.get(qid, {}).get("label", "")
        if l3 in c3:
            c3[l3] += 1
        if l4 in c4:
            c4[l4] += 1

    print()
    print(f"  {'':>12} {'bueno':>7} {'tang':>7} {'malo':>7}")
    print(f"  {'─' * 35}")
    print(f"  {'v3b (BGE)':>12} {c3['b']:>4}/21  {c3['t']:>4}/21  {c3['m']:>4}/21")
    print(f"  {'v4.2 (hyb)':>12} {c4['b']:>4}/21  {c4['t']:>4}/21  {c4['m']:>4}/21")

    # --- Gains y losses ---
    print()
    if gains:
        print(f"  Gains (+bueno): {', '.join(f'Q{q}' for q in gains)}")
        for qid in gains:
            hr = hybrid.get(qid, {})
            print(f"    Q{qid}: {v3b[qid]['label']}({short_source(hr.get('dense_source', ''))}) "
                  f"-> b({short_source(hr.get('chunk_source', ''))})  [{hr.get('change_type', '')}]")
    else:
        print("  Gains: ninguno")

    if losses:
        print(f"  Losses (-bueno): {', '.join(f'Q{q}' for q in losses)}")
        for qid in losses:
            hr = hybrid.get(qid, {})
            print(f"    Q{qid}: b({short_source(hr.get('dense_source', ''))}) "
                  f"-> {v42[qid]['label']}({short_source(hr.get('chunk_source', ''))})  [{hr.get('change_type', '')}]")
    else:
        print("  Losses: ninguno")

    if downgrades:
        print(f"  Downgrades (t->m): {', '.join(f'Q{q}' for q in downgrades)}")

    net = len(gains) - len(losses)
    print(f"\n  Neto: {c3['b']}/21 -> {c4['b']}/21 (gains={len(gains)}, losses={len(losses)}, net={net:+d})")

    # --- Prediccion ---
    pred = args.predict
    actual = c4["b"]
    delta = actual - pred
    print(f"\n  Prediccion: {pred}/21")
    print(f"  Real:       {actual}/21")
    print(f"  Error:      {delta:+d}")
    print()


if __name__ == "__main__":
    main()
