import json
import time
from pathlib import Path
from finder_bm25 import (
    load_chunks, build_bm25_index, search, tokenize, tokenize_smart, DOCS_DIR,
)

SCRIPT_DIR = Path(__file__).parent
EVAL_DATASET = SCRIPT_DIR / "eval_dataset.jsonl"
BM25_RESULTS_DUMB = SCRIPT_DIR / "bm25_results_dumb.jsonl"
BM25_RESULTS_SMART = SCRIPT_DIR / "bm25_results_smart.jsonl"
V1_RESULTS = SCRIPT_DIR.parent / "v1" / "eval_results.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_source(s: str) -> str:
    """Extrae solo el path relativo desde docs-source/ para comparar."""
    marker = "docs-source/"
    idx = s.find(marker)
    if idx != -1:
        return s[idx:]
    return s


def run_bm25(chunks, dataset, tokenizer, label):
    """Construye índice y corre las 21 queries con un tokenizer dado."""
    t0 = time.time()
    index = build_bm25_index(chunks, tokenizer=tokenizer)
    t_index = time.time() - t0
    print(f"[{label}] Indexación: {t_index:.2f}s")

    results = []
    for entry in dataset:
        result = search(entry["query"], chunks, index, tokenizer=tokenizer)
        results.append({
            "id": entry["id"],
            "query": entry["query"],
            "expected": entry["expected"],
            "category": entry["category"],
            "bm25_score": result["chunk_score"],
            "bm25_source": result["chunk_source"],
            "bm25_chunk_text": result["chunk_text"],
        })
    return results


def save_jsonl(data, path):
    with open(path, "w") as f:
        for r in data:
            f.write(json.dumps(r) + "\n")


def main():
    chunks = load_chunks(DOCS_DIR)
    print(f"Corpus: {len(chunks)} chunks\n")

    dataset = load_jsonl(EVAL_DATASET)
    v1_results = {r["id"]: r for r in load_jsonl(V1_RESULTS)}

    # Correr ambos tokenizers
    dumb = run_bm25(chunks, dataset, tokenize, "dumb")
    smart = run_bm25(chunks, dataset, tokenize_smart, "smart")

    save_jsonl(dumb, BM25_RESULTS_DUMB)
    save_jsonl(smart, BM25_RESULTS_SMART)
    print(f"\nGuardados: {BM25_RESULTS_DUMB.name}, {BM25_RESULTS_SMART.name}\n")

    # Tabla de 3 columnas
    dumb_by_id = {r["id"]: r for r in dumb}
    smart_by_id = {r["id"]: r for r in smart}

    header = (
        f"{'ID':>3} {'CAT':<17} "
        f"{'EMB_SRC':<30} {'DUMB_SRC':<30} {'SMART_SRC':<30} "
        f"{'D→S':>3} "
        f"{'QUERY':<50}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    dumb_changed = 0
    smart_changed = 0
    dumb_vs_smart_changed = 0

    for entry in dataset:
        qid = entry["id"]
        v1 = v1_results.get(qid, {})
        emb_src = normalize_source(v1.get("chunk_source", ""))
        dumb_src = normalize_source(dumb_by_id[qid]["bm25_source"])
        smart_src = normalize_source(smart_by_id[qid]["bm25_source"])

        # Shorten paths for display
        def short(s):
            return s.replace("docs-source/", "").replace("user_guide/torch_compiler/", "tc/")[:28]

        d_vs_s = "Y" if dumb_src != smart_src else "N"
        if emb_src != dumb_src:
            dumb_changed += 1
        if emb_src != smart_src:
            smart_changed += 1
        if dumb_src != smart_src:
            dumb_vs_smart_changed += 1

        print(
            f"{qid:>3} {entry['category']:<17} "
            f"{short(emb_src):<30} {short(dumb_src):<30} {short(smart_src):<30} "
            f"{d_vs_s:>3} "
            f"{entry['query']:<50}"
        )

    print(sep)
    print(f"\nvs embeddings — dumb changed: {dumb_changed}/21 | smart changed: {smart_changed}/21")
    print(f"dumb vs smart differ: {dumb_vs_smart_changed}/21\n")

    # Scores comparison for queries of interest
    focus_ids = [2, 11, 13, 15, 20]
    print(f"{'='*90}")
    print("SCORES DETALLE — queries de interés")
    print(f"{'='*90}")
    print(f"{'ID':>3} {'EMB':>8} {'DUMB':>8} {'SMART':>8}  {'QUERY':<50}")
    print("-" * 90)
    for qid in focus_ids:
        v1 = v1_results.get(qid, {})
        print(
            f"{qid:>3} "
            f"{v1.get('chunk_score', 0):>8.4f} "
            f"{dumb_by_id[qid]['bm25_score']:>8.4f} "
            f"{smart_by_id[qid]['bm25_score']:>8.4f}  "
            f"{dumb_by_id[qid]['query']:<50}"
        )
    print()

    # Detail where smart differs from dumb
    print(f"{'='*90}")
    print("DETALLE: queries donde smart tokenizer cambió el chunk vs dumb")
    print(f"{'='*90}\n")
    for entry in dataset:
        qid = entry["id"]
        d = dumb_by_id[qid]
        s = smart_by_id[qid]
        dumb_src = normalize_source(d["bm25_source"])
        smart_src = normalize_source(s["bm25_source"])

        if dumb_src != smart_src:
            v1 = v1_results.get(qid, {})
            emb_src = normalize_source(v1.get("chunk_source", ""))
            print(f"--- Query {qid}: {entry['query']} ---")
            print(f"  Embeddings -> {emb_src}")
            print(f"  BM25 dumb  -> {dumb_src}  (score: {d['bm25_score']:.2f})")
            print(f"  BM25 smart -> {smart_src}  (score: {s['bm25_score']:.2f})")
            print()


if __name__ == "__main__":
    main()
