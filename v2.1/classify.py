"""
classify.py — Clasificación ciega de chunks de retrieval.

Muestra cada query y su chunk ganador, uno a la vez.
Pide b/t/m en la terminal. No muestra contadores, predicciones,
ni chunks de otros sistemas. Escribe etiquetas a CSV.

Uso:
    python v2.1/classify.py results.jsonl -o labels.csv
    python v2.1/classify.py results.jsonl                    # default: labels_<stem>.csv

El JSONL puede tener cualquiera de estos formatos:
    - chunk_source / chunk_text           (v1 eval_results.jsonl)
    - bm25_source / bm25_chunk_text       (bm25_results_smart.jsonl)
    - hybrid_source / hybrid_chunk_text   (hybrid_results.jsonl)
    - Cualquier par *_source / *_chunk_text

Si el CSV de salida ya existe y tiene etiquetas parciales, retoma
desde donde quedó (no re-pregunta las ya clasificadas).
"""

import argparse
import csv
import json
import os
import sys
import textwrap
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["id"])
    return records


def detect_keys(record):
    """Detecta las keys de source y chunk_text en el record."""
    source_key = None
    text_key = None
    for k in record:
        if k.endswith("_source") or k == "chunk_source":
            source_key = k
        if k.endswith("_chunk_text") or k == "chunk_text":
            text_key = k
    if not source_key or not text_key:
        # Fallback: try exact names
        if "chunk_source" in record:
            source_key = "chunk_source"
        if "chunk_text" in record:
            text_key = "chunk_text"
    return source_key, text_key


def short_source(path_str):
    """Extrae ruta relativa desde docs-source/."""
    if not path_str:
        return "?"
    p = Path(path_str)
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "docs-source" and i + 1 < len(parts):
            return "/".join(parts[i + 1:])
    return p.name


def load_existing_labels(csv_path):
    """Carga etiquetas ya clasificadas de un CSV existente."""
    labels = {}
    if not os.path.exists(csv_path):
        return labels
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            qid = int(row["query_id"])
            label = row.get("label", "").strip()
            if label in ("b", "t", "m"):
                labels[qid] = label
    return labels


def write_csv(csv_path, rows):
    """Escribe todas las filas al CSV (reescritura completa)."""
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query", "source", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def wrap_text(text, width=78, indent="  "):
    """Wrap text para display en terminal."""
    # Limpiar whitespace interno
    clean = " ".join(text.split())
    return textwrap.fill(clean, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def main():
    parser = argparse.ArgumentParser(
        description="Clasificación ciega de chunks: b(ueno) / t(angencial) / m(alo)")
    parser.add_argument("jsonl", help="Archivo JSONL de resultados")
    parser.add_argument("-o", "--output", help="CSV de salida (default: labels_<stem>.csv)")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} no existe", file=sys.stderr)
        sys.exit(1)

    # Default output path
    if args.output:
        csv_path = Path(args.output)
    else:
        csv_path = jsonl_path.parent / f"labels_{jsonl_path.stem}.csv"

    records = load_jsonl(jsonl_path)
    if not records:
        print("Error: JSONL vacío", file=sys.stderr)
        sys.exit(1)

    source_key, text_key = detect_keys(records[0])
    if not source_key or not text_key:
        print(f"Error: no pude detectar keys de source/chunk_text en {jsonl_path}",
              file=sys.stderr)
        print(f"  Keys encontradas: {list(records[0].keys())}", file=sys.stderr)
        sys.exit(1)

    # Cargar etiquetas existentes para retomar
    existing = load_existing_labels(csv_path)
    if existing:
        print("Retomando...")

    # Construir filas (mantener orden y rellenar con existentes)
    rows = []
    pending = []
    for rec in records:
        qid = rec["id"]
        query = rec["query"]
        source = short_source(rec.get(source_key, ""))
        label = existing.get(qid, "")
        rows.append({"query_id": qid, "query": query, "source": source, "label": label})
        if not label:
            pending.append((len(rows) - 1, rec))

    if not pending:
        print(f"Todas las {len(records)} queries ya están clasificadas en {csv_path}")
        sys.exit(0)

    total = len(records)
    print(f"Criterio: b(ueno) = el chunk RESPONDE la pregunta, no solo la menciona")
    print(f"          t(angencial) = relacionado al tema pero no responde")
    print(f"          m(alo) = irrelevante o completamente fuera de tema")
    print()
    print(f"  Output: {csv_path}")
    print()
    print("─" * 78)

    try:
        for idx, (row_idx, rec) in enumerate(pending):
            qid = rec["id"]
            query = rec["query"]
            source = short_source(rec.get(source_key, ""))
            chunk_text = rec.get(text_key, "(sin texto)")

            # Número de progreso sin contadores de b/t/m
            # remaining = len(pending) - idx # intencionalmente no se muestra — romperia la clasificación ciega
            os.system('clear')

            print()
            print(f"  QUERY: {query}")
            print()
            print(f"  SOURCE: {source}")
            print()
            print(f"  CHUNK:")
            print(wrap_text(chunk_text, width=76, indent="    "))
            print()

            # Pedir clasificación
            while True:
                try:
                    resp = input("  b / t / m ? ").strip().lower()
                except EOFError:
                    print("\n\nInterrumpido. Guardando progreso...")
                    write_csv(csv_path, rows)
                    print(f"Guardado en {csv_path}")
                    sys.exit(0)

                if resp in ("b", "t", "m"):
                    break
                if resp in ("q", "quit", "exit"):
                    print("\nGuardando progreso...")
                    write_csv(csv_path, rows)
                    print(f"Guardado en {csv_path}")
                    sys.exit(0)
                print("  → Ingresá b, t, o m (o q para salir y guardar)")

            rows[row_idx]["label"] = resp
            # Guardar después de cada clasificación (crash-safe)
            write_csv(csv_path, rows)

            print("─" * 78)

    except KeyboardInterrupt:
        print("\n\nInterrumpido. Guardando progreso...")
        write_csv(csv_path, rows)
        print(f"Guardado en {csv_path}")
        sys.exit(0)

    print()
    print(f"Clasificación completa. {len(records)} etiquetas guardadas en {csv_path}")
    print(f"Corré count.py para ver los conteos.")


if __name__ == "__main__":
    main()
