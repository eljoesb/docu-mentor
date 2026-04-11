"""
measure_hybrid.py — Hybrid retrieval BGE + BM25 con weighted RRF.

Dense: BAAI/bge-base-en-v1.5 (reutiliza cache de v3)
Sparse: BM25Okapi con tokenize_smart (rompe por puntos/guiones, de v2)
Fusion: Weighted RRF (alpha_dense=0.7, alpha_sparse=0.3, k=60)

Produce hybrid_results.jsonl compatible con classify.py --axis retrieval.
Para cada query guarda: chunk ganador, rrf_score, dense_rank, sparse_rank,
change_type, y las fuentes de BGE-only y BM25-only para comparacion.
"""

import json
import re
import time
import hashlib
import pickle
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

DOCS_DIR = BASE_DIR / "docs-source"
CACHE_PATH = BASE_DIR / "v3" / "embeddings_cache_bge_base.pkl"
DATASET_PATH = BASE_DIR / "v3" / "eval_dataset.jsonl"
RESULTS_PATH = SCRIPT_DIR / "hybrid_results.jsonl"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Weighted RRF parameters (prediction.md)
RRF_K = 60
ALPHA_DENSE = 0.7
ALPHA_SPARSE = 0.3
TOP_N = 50


# --- Chunking (identico a v3/v2/v1) ---

def load_or_build_chunks(docs_dir, min_palabras=50):
    archivos = list(docs_dir.rglob("*.rst")) + list(docs_dir.rglob("*.md"))
    chunks = []
    for archivo in archivos:
        texto = archivo.read_text(encoding="utf-8", errors="ignore")
        parrafos = texto.split("\n\n")
        buffer = ""
        for parrafo in parrafos:
            parrafo = parrafo.strip()
            if not parrafo:
                continue
            buffer = buffer + "\n\n" + parrafo if buffer else parrafo
            if len(buffer.split()) >= min_palabras:
                chunks.append({"source": str(archivo), "text": buffer})
                buffer = ""
        if buffer.strip():
            if chunks and chunks[-1]["source"] == str(archivo):
                chunks[-1]["text"] += "\n\n" + buffer
            else:
                chunks.append({"source": str(archivo), "text": buffer})
    return chunks


def corpus_hash(chunks):
    h = hashlib.md5()
    for c in chunks:
        h.update(c["source"].encode())
        h.update(c["text"][:100].encode())
    return h.hexdigest()


def load_or_build_embeddings(chunks, model, cache_path):
    current_hash = corpus_hash(chunks)
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("hash") == current_hash:
            print(f"  Cache valido: {cache_path}")
            return cache["embeddings"]
        print("  Cache invalido (hash cambio), re-encoding...")

    print(f"  Encoding {len(chunks)} chunks...")
    t0 = time.time()
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)
    print(f"  Encoding completado en {time.time() - t0:.1f}s")

    with open(cache_path, "wb") as f:
        pickle.dump({"hash": current_hash, "embeddings": embeddings}, f)
    return embeddings


# --- Tokenizer smart (de v2/finder_bm25.py) ---

def tokenize_smart(text):
    text = text.lower()
    text = re.sub(r"[.\-_()/\\\[\]{}'\"`]", " ", text)
    return [t for t in text.split() if t]


# --- Retrieval ---

def get_dense_ranking(query, embeddings, model, top_n=TOP_N):
    query_emb = model.encode([QUERY_PREFIX + query])
    scores = cosine_similarity(query_emb, embeddings)[0]
    ranking = np.argsort(scores)[::-1][:top_n].tolist()
    return ranking, scores


def get_sparse_ranking(query, bm25_index, top_n=TOP_N):
    tokens = tokenize_smart(query)
    scores = bm25_index.get_scores(tokens)
    ranking = np.argsort(scores)[::-1][:top_n].tolist()
    return ranking, scores


def weighted_rrf_fuse(dense_ranking, sparse_ranking,
                      k=RRF_K, alpha_d=ALPHA_DENSE, alpha_s=ALPHA_SPARSE):
    scores = {}
    for rank, idx in enumerate(dense_ranking):
        scores[idx] = scores.get(idx, 0) + alpha_d / (k + rank)
    for rank, idx in enumerate(sparse_ranking):
        scores[idx] = scores.get(idx, 0) + alpha_s / (k + rank)
    return scores


def classify_change(winner_idx, dense_top1, sparse_top1):
    if winner_idx == dense_top1 and winner_idx == sparse_top1:
        return "consensus"
    if winner_idx == dense_top1:
        return "dense_wins"
    if winner_idx == sparse_top1:
        return "sparse_wins"
    return "compromise"


# --- Helpers ---

def short_source(path_str):
    if not path_str:
        return "?"
    p = Path(path_str)
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "docs-source" and i + 1 < len(parts):
            return "/".join(parts[i + 1:])
    return p.name


def load_dataset(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- Main ---

def main():
    print("=" * 70)
    print("v4.2 measure_hybrid.py — Weighted RRF (BGE + BM25)")
    print(f"  Dense:  {MODEL_NAME} (alpha={ALPHA_DENSE})")
    print(f"  Sparse: BM25-smart (alpha={ALPHA_SPARSE})")
    print(f"  RRF k={RRF_K}, top_n={TOP_N}")
    print("=" * 70)

    print(f"\nCargando modelo {MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Modelo cargado en {time.time() - t0:.1f}s")

    print("\nPreparando corpus...")
    chunks = load_or_build_chunks(DOCS_DIR)
    print(f"  {len(chunks)} chunks")
    embeddings = load_or_build_embeddings(chunks, model, CACHE_PATH)

    print("\nConstruyendo indice BM25-smart...")
    t0 = time.time()
    corpus_tokens = [tokenize_smart(c["text"]) for c in chunks]
    bm25_index = BM25Okapi(corpus_tokens)
    print(f"  Indice construido en {time.time() - t0:.1f}s")

    dataset = load_dataset(DATASET_PATH)
    print(f"\nDataset: {len(dataset)} queries")

    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    print(f"\n{'=' * 70}")
    print("Corriendo hybrid retrieval...")
    print(f"{'=' * 70}\n")

    results = []
    diag_data = {}

    for entry in dataset:
        qid = entry["id"]
        query = entry["query"]

        t0 = time.time()
        dense_ranking, dense_scores = get_dense_ranking(query, embeddings, model)
        sparse_ranking, sparse_scores = get_sparse_ranking(query, bm25_index)
        fused = weighted_rrf_fuse(dense_ranking, sparse_ranking)
        winner_idx = max(fused, key=fused.get)
        t_search = time.time() - t0

        dense_top1 = dense_ranking[0]
        sparse_top1 = sparse_ranking[0]
        change_type = classify_change(winner_idx, dense_top1, sparse_top1)

        d_rank = dense_ranking.index(winner_idx) if winner_idx in dense_ranking else None
        s_rank = sparse_ranking.index(winner_idx) if winner_idx in sparse_ranking else None

        record = {
            "id": qid,
            "query": query,
            "category": entry["category"],
            "expected": entry["expected"],
            "chunk_source": chunks[winner_idx]["source"],
            "chunk_text": chunks[winner_idx]["text"],
            "rrf_score": round(fused[winner_idx], 6),
            "dense_rank": d_rank,
            "sparse_rank": s_rank,
            "change_type": change_type,
            "dense_source": chunks[dense_top1]["source"],
            "sparse_source": chunks[sparse_top1]["source"],
            "t_search": round(t_search, 4),
        }
        results.append(record)

        # Store diagnostic data for Q13, Q19
        if qid in (13, 19):
            diag_data[qid] = {
                "dense_ranking": dense_ranking[:10],
                "sparse_ranking": sparse_ranking[:10],
                "dense_scores": dense_scores,
                "sparse_scores": sparse_scores,
                "fused": fused,
            }

        hyb = short_source(chunks[winner_idx]["source"])
        den = short_source(chunks[dense_top1]["source"])
        marker = " << CHANGED" if winner_idx != dense_top1 else ""

        print(f"  Q{qid:>2}: {change_type:<12} "
              f"D:{d_rank if d_rank is not None else '-':>2} "
              f"S:{s_rank if s_rank is not None else '-':>2}  "
              f"rrf={fused[winner_idx]:.5f}  "
              f"hyb={hyb:<30} den={den:<30}{marker}")

    # Save
    with open(RESULTS_PATH, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    changed = sum(1 for r in results
                  if r["chunk_source"] != r["dense_source"])
    type_counts = {}
    for r in results:
        ct = r["change_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1

    print(f"\n{'=' * 70}")
    print(f"Resultados guardados en: {RESULTS_PATH}")
    print(f"Hybrid vs BGE-puro: {changed}/21 chunks cambiaron")
    print(f"Change types: {type_counts}")

    # --- Diagnostic: Q13 and Q19 ---
    print(f"\n{'=' * 70}")
    print("DIAGNOSTICO: Q13 y Q19 — verificacion de hipotesis mecanica")
    print(f"{'=' * 70}")

    for qid in [13, 19]:
        r = next(x for x in results if x["id"] == qid)
        diag = diag_data[qid]

        print(f"\n--- Q{qid}: {r['query']} ---")
        print(f"  Resultado:  {r['change_type']}")
        print(f"  BGE top-1:    {short_source(r['dense_source'])}")
        print(f"  BM25 top-1:   {short_source(r['sparse_source'])}")
        print(f"  Hybrid top-1: {short_source(r['chunk_source'])}")
        print(f"  Winner ranks: dense={r['dense_rank']}, sparse={r['sparse_rank']}")

        print(f"\n  BGE top-10:")
        for rank, idx in enumerate(diag["dense_ranking"]):
            src = short_source(chunks[idx]["source"])
            score = float(diag["dense_scores"][idx])
            flag = " <-- hybrid winner" if idx == max(diag["fused"], key=diag["fused"].get) else ""
            print(f"    {rank:>2}: {src:<40} cosine={score:.4f}{flag}")

        print(f"\n  BM25-smart top-10:")
        for rank, idx in enumerate(diag["sparse_ranking"]):
            src = short_source(chunks[idx]["source"])
            score = float(diag["sparse_scores"][idx])
            flag = " <-- hybrid winner" if idx == max(diag["fused"], key=diag["fused"].get) else ""
            print(f"    {rank:>2}: {src:<40} bm25={score:.4f}{flag}")

        # Show RRF scores of top contenders
        print(f"\n  Top-5 RRF scores:")
        sorted_fused = sorted(diag["fused"].items(), key=lambda x: x[1], reverse=True)[:5]
        for fuse_rank, (idx, score) in enumerate(sorted_fused):
            src = short_source(chunks[idx]["source"])
            d_r = diag["dense_ranking"].index(idx) if idx in diag["dense_ranking"] else ">10"
            s_r = diag["sparse_ranking"].index(idx) if idx in diag["sparse_ranking"] else ">10"
            print(f"    {fuse_rank}: {src:<40} rrf={score:.6f}  (D:{d_r} S:{s_r})")

        print(f"\n  Chunk ganador (300 chars):")
        print(f"    {r['chunk_text'][:300]}")

    print(f"\n{'=' * 70}")
    print("Listo. Clasifica con: python v4.2/classify.py v4.2/hybrid_results.jsonl --axis retrieval")
    print("IMPORTANTE: clasifica con cabeza fresca, no ahora si es tarde.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
