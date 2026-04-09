import json
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from finder_bm25 import load_chunks, build_bm25_index, tokenize_smart, DOCS_DIR
from finder_hybrid import (
    load_embeddings_cache, hybrid_search, get_dense_ranking, get_sparse_ranking,
    CACHE_PATH,
)

SCRIPT_DIR = Path(__file__).parent
EVAL_DATASET = SCRIPT_DIR / "eval_dataset.jsonl"
HYBRID_RESULTS = SCRIPT_DIR / "hybrid_results.jsonl"
V1_RESULTS = SCRIPT_DIR.parent / "v1" / "eval_results.jsonl"
BM25_SMART_RESULTS = SCRIPT_DIR / "bm25_results_smart.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_source(s: str) -> str:
    marker = "docs-source/"
    idx = s.find(marker)
    if idx != -1:
        return s[idx:]
    return s


def short_source(s: str) -> str:
    return (normalize_source(s)
            .replace("docs-source/", "")
            .replace("user_guide/torch_compiler/", "tc/"))[:28]


def classify_change(dense_rank, sparse_rank, dense_top1, sparse_top1, winner_idx):
    """Clasifica cómo RRF llegó a su decisión."""
    if winner_idx == dense_top1 and winner_idx == sparse_top1:
        return "consensus"
    if winner_idx == dense_top1:
        return "dense_wins"
    if winner_idx == sparse_top1:
        return "sparse_wins"
    return "compromise"


def main():
    # Cargar todo
    chunks = load_chunks(DOCS_DIR)
    embeddings_chunks = load_embeddings_cache(CACHE_PATH, chunks)
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    bm25_index = build_bm25_index(chunks, tokenizer=tokenize_smart)
    print(f"Corpus: {len(chunks)} chunks\n")

    dataset = load_jsonl(EVAL_DATASET)
    v1_results = {r["id"]: r for r in load_jsonl(V1_RESULTS)}
    bm25_smart = {r["id"]: r for r in load_jsonl(BM25_SMART_RESULTS)}

    # Correr hybrid sobre las 21 queries
    hybrid_results = []
    for entry in dataset:
        qid = entry["id"]
        query = entry["query"]

        t0 = time.time()
        # Get individual top-1s for classification
        dense_ranking = get_dense_ranking(query, chunks, embeddings_chunks, modelo)
        sparse_ranking = get_sparse_ranking(query, chunks, bm25_index)
        dense_top1 = dense_ranking[0]
        sparse_top1 = sparse_ranking[0]

        result = hybrid_search(query, chunks, embeddings_chunks, modelo, bm25_index)
        t_search = time.time() - t0

        change_type = classify_change(
            result["dense_rank"], result["sparse_rank"],
            dense_top1, sparse_top1, result["chunk_idx"],
        )

        record = {
            "id": qid,
            "query": query,
            "expected": entry["expected"],
            "category": entry["category"],
            "hybrid_source": result["chunk_source"],
            "hybrid_chunk_text": result["chunk_text"],
            "rrf_score": result["rrf_score"],
            "dense_rank": result["dense_rank"],
            "sparse_rank": result["sparse_rank"],
            "change_type": change_type,
            "t_search": round(t_search, 4),
        }
        hybrid_results.append(record)

    # Guardar
    with open(HYBRID_RESULTS, "w") as f:
        for r in hybrid_results:
            f.write(json.dumps(r) + "\n")
    print(f"Guardados en {HYBRID_RESULTS}\n")

    # Tabla principal
    header = (
        f"{'ID':>3} {'CAT':<17} "
        f"{'EMB_SRC':<30} {'BM25S_SRC':<30} {'HYBRID_SRC':<30} "
        f"{'D':>2} {'S':>2} "
        f"{'TYPE':<13} "
        f"{'QUERY':<50}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    type_counts = {}
    for r in hybrid_results:
        qid = r["id"]
        v1 = v1_results.get(qid, {})
        bm25s = bm25_smart.get(qid, {})

        emb_src = short_source(v1.get("chunk_source", ""))
        bm25s_src = short_source(bm25s.get("bm25_source", ""))
        hyb_src = short_source(r["hybrid_source"])

        dr = r["dense_rank"] if r["dense_rank"] is not None else "-"
        sr = r["sparse_rank"] if r["sparse_rank"] is not None else "-"

        ct = r["change_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1

        print(
            f"{qid:>3} {r['category']:<17} "
            f"{emb_src:<30} {bm25s_src:<30} {hyb_src:<30} "
            f"{str(dr):>2} {str(sr):>2} "
            f"{ct:<13} "
            f"{r['query']:<50}"
        )

    print(sep)
    print(f"\nCHANGE_TYPE counts: {type_counts}")

    # Hybrid vs embeddings cambios
    changed = sum(
        1 for r in hybrid_results
        if normalize_source(r["hybrid_source"]) != normalize_source(v1_results[r["id"]].get("chunk_source", ""))
    )
    same = len(hybrid_results) - changed
    print(f"Hybrid vs embeddings: changed {changed}/21, same {same}/21\n")

    # Detalle de las 5 queries de interés
    focus = [11, 7, 13, 2, 18]
    print(f"{'='*90}")
    print("DETALLE: 5 queries de interés")
    print(f"{'='*90}\n")
    for qid in focus:
        r = next(x for x in hybrid_results if x["id"] == qid)
        v1 = v1_results.get(qid, {})
        bm25s = bm25_smart.get(qid, {})
        print(f"--- Q{qid}: {r['query']} (expected: {r['expected']}, cat: {r['category']}) ---")
        print(f"  Embeddings -> {short_source(v1.get('chunk_source', ''))}")
        print(f"  BM25-smart -> {short_source(bm25s.get('bm25_source', ''))}")
        print(f"  Hybrid     -> {short_source(r['hybrid_source'])}  (D:{r['dense_rank']} S:{r['sparse_rank']} type:{r['change_type']})")
        print(f"  Hybrid chunk (first 300 chars):")
        print(f"    {r['hybrid_chunk_text'][:300]}")
        print()


if __name__ == "__main__":
    main()
