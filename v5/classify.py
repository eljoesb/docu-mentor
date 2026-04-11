"""
classify.py — Clasificacion ciega por eje para v5.

Igual que v3/classify.py pero:
  - Eje generation tiene 4 etiquetas: f/e/x/i (agrega 'empty')
  - Lee campos de v5 (original_chunk, retry_chunk, final_response, etc.)
  - Usa el estado final (retry si existe, original si no)

Uso:
    python v5/classify.py v5/pipeline_results_v5.jsonl --axis retrieval
    python v5/classify.py v5/pipeline_results_v5.jsonl --axis generation
    python v5/classify.py v5/pipeline_results_v5.jsonl --axis utility
"""

import argparse
import csv
import json
import os
import sys
import textwrap
from pathlib import Path

AXES = {
    "retrieval": {
        "labels": {"b": "bueno", "t": "tangencial", "m": "malo"},
        "prompt": "b / t / m ? ",
        "criteria": (
            "b(ueno) = el chunk RESPONDE la pregunta, no solo la menciona\n"
            "          t(angencial) = relacionado al tema pero no responde\n"
            "          m(alo) = irrelevante o completamente fuera de tema"
        ),
    },
    "generation": {
        "labels": {"f": "fiel", "e": "empty", "x": "mixto", "i": "inventado"},
        "prompt": "f / e / x / i ? ",
        "criteria": (
            "f(iel) = la respuesta usa SOLO info del chunk, no agrega nada\n"
            "          e(mpty) = respuesta vacia/generica que no usa el chunk ni inventa\n"
            "          x(mixto) = mezcla info del chunk con info inventada\n"
            "          i(nventado) = ignora el chunk o inventa informacion"
        ),
    },
    "utility": {
        "labels": {"u": "util", "p": "parcial", "d": "danino"},
        "prompt": "u / p / d ? ",
        "criteria": (
            "u(til) = un usuario quedaria bien servido con esta respuesta\n"
            "          p(arcial) = algo util pero incompleta o vaga\n"
            "          d(anino) = incorrecta, enganiosa, o peor que no responder"
        ),
    },
}


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["id"])
    return records


def normalize_record(rec):
    """Convierte campos v5 a los nombres que espera display_for_axis."""
    # Usar estado final: retry si existe, original si no
    chunk_source = rec.get("retry_chunk_source") or rec["original_chunk_source"]
    chunk_text = rec.get("retry_chunk") or rec["original_chunk"]
    llm_response = rec["final_response"]
    return {
        "id": rec["id"],
        "query": rec["query"],
        "chunk_source": chunk_source,
        "chunk_text": chunk_text,
        "llm_response": llm_response,
        "retried": rec.get("retried", False),
    }


def short_source(path_str):
    if not path_str:
        return "?"
    p = Path(path_str)
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "docs-source" and i + 1 < len(parts):
            return "/".join(parts[i + 1:])
    return p.name


def load_existing_labels(csv_path, valid_labels):
    labels = {}
    if not os.path.exists(csv_path):
        return labels
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["query_id"])
            label = row.get("label", "").strip()
            if label in valid_labels:
                labels[qid] = label
    return labels


def write_csv(csv_path, rows):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query", "source", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def wrap_text(text, width=78, indent="  "):
    clean = " ".join(text.split())
    return textwrap.fill(clean, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def display_for_axis(axis, rec):
    """Muestra solo la info relevante para el eje."""
    query = rec["query"]
    source = short_source(rec.get("chunk_source", ""))
    chunk = rec.get("chunk_text", "(sin texto)")
    llm = rec.get("llm_response", "(sin respuesta)")

    if axis == "retrieval":
        print(f"  QUERY: {query}")
        print()
        print(f"  SOURCE: {source}")
        print()
        print(f"  CHUNK:")
        print(wrap_text(chunk, width=76, indent="    "))

    elif axis == "generation":
        print(f"  SOURCE: {source}")
        print()
        print(f"  CHUNK:")
        print(wrap_text(chunk, width=76, indent="    "))
        print()
        print(f"  LLM RESPONSE:")
        print(wrap_text(llm, width=76, indent="    "))

    elif axis == "utility":
        print(f"  QUERY: {query}")
        print()
        print(f"  LLM RESPONSE:")
        print(wrap_text(llm, width=76, indent="    "))


def main():
    parser = argparse.ArgumentParser(
        description="Clasificacion ciega por eje: retrieval / generation / utility")
    parser.add_argument("jsonl", help="Archivo JSONL de resultados del pipeline")
    parser.add_argument("--axis", required=True, choices=AXES.keys(),
                        help="Eje a clasificar")
    parser.add_argument("-o", "--output", help="CSV de salida (default: labels_<axis>_v5.csv)")
    args = parser.parse_args()

    axis = args.axis
    axis_config = AXES[axis]
    valid_labels = set(axis_config["labels"].keys())

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} no existe", file=sys.stderr)
        sys.exit(1)

    if args.output:
        csv_path = Path(args.output)
    else:
        csv_path = jsonl_path.parent / f"labels_{axis}_v5.csv"

    raw_records = load_jsonl(jsonl_path)
    records = [normalize_record(r) for r in raw_records]
    if not records:
        print("Error: JSONL vacio", file=sys.stderr)
        sys.exit(1)

    existing = load_existing_labels(csv_path, valid_labels)
    if existing:
        print("Retomando...")

    rows = []
    pending = []
    for rec in records:
        qid = rec["id"]
        query = rec["query"]
        source = short_source(rec.get("chunk_source", ""))
        label = existing.get(qid, "")
        rows.append({"query_id": qid, "query": query, "source": source, "label": label})
        if not label:
            pending.append((len(rows) - 1, rec))

    if not pending:
        print(f"Todas las {len(records)} queries ya estan clasificadas en {csv_path}")
        sys.exit(0)

    label_names = " / ".join(f"{k}({v})" for k, v in axis_config["labels"].items())
    print(f"\nEje: {axis.upper()}")
    print(f"Etiquetas: {label_names}")
    print()
    print(f"Criterio:")
    print(f"  {axis_config['criteria']}")
    print()
    print(f"  Output: {csv_path}")
    print()
    print("-" * 78)

    try:
        for idx, (row_idx, rec) in enumerate(pending):
            os.system('clear')

            print()
            print(f"  [{axis.upper()}]  ({idx+1}/{len(pending)})")
            if rec.get("retried"):
                print(f"  [RETRIED]")
            print()
            display_for_axis(axis, rec)
            print()

            while True:
                try:
                    resp = input(f"  {axis_config['prompt']}").strip().lower()
                except EOFError:
                    print("\n\nInterrumpido. Guardando progreso...")
                    write_csv(csv_path, rows)
                    print(f"Guardado en {csv_path}")
                    sys.exit(0)

                if resp in valid_labels:
                    break
                if resp in ("q", "quit", "exit"):
                    print("\nGuardando progreso...")
                    write_csv(csv_path, rows)
                    print(f"Guardado en {csv_path}")
                    sys.exit(0)
                valid_str = ", ".join(valid_labels)
                print(f"  -> Ingresa {valid_str} (o q para salir y guardar)")

            rows[row_idx]["label"] = resp
            write_csv(csv_path, rows)

            print("-" * 78)

    except KeyboardInterrupt:
        print("\n\nInterrumpido. Guardando progreso...")
        write_csv(csv_path, rows)
        print(f"Guardado en {csv_path}")
        sys.exit(0)

    print()
    print(f"Clasificacion {axis} completa. {len(records)} etiquetas en {csv_path}")


if __name__ == "__main__":
    main()
