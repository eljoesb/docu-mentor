import sys
import time
import hashlib
import pickle
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from finder_bm25 import load_chunks, build_bm25_index, tokenize_smart, DOCS_DIR

SCRIPT_DIR = Path(__file__).parent
CACHE_PATH = SCRIPT_DIR.parent / "v1" / "embeddings_cache.pkl"


def corpus_hash(chunks: list[dict]) -> str:
    """Mismo hash que v1/documentor.py para validar el caché."""
    h = hashlib.md5()
    for c in chunks:
        h.update(c["source"].encode())
        h.update(c["text"][:100].encode())
    return h.hexdigest()


def load_embeddings_cache(cache_path: Path, chunks: list[dict]):
    """Carga embeddings desde el caché de v1. Falla si no existe o hash no coincide."""
    if not cache_path.exists():
        raise FileNotFoundError(f"Caché no encontrado: {cache_path}. Corré v1/documentor.py primero.")
    current_hash = corpus_hash(chunks)
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    if cache.get("hash") != current_hash:
        raise ValueError(f"Hash del corpus no coincide con el caché. Regenerá corriendo v1/documentor.py.")
    return cache["embeddings"]


def get_dense_ranking(query: str, chunks, embeddings_chunks, modelo, top_n: int = 50) -> list[int]:
    """Top-N chunk indices según embeddings, del más relevante al menos."""
    embedding_query = modelo.encode([query])
    scores = cosine_similarity(embedding_query, embeddings_chunks)[0]
    top_indices = np.argsort(scores)[::-1][:top_n]
    return top_indices.tolist()


def get_sparse_ranking(query: str, chunks, bm25_index, top_n: int = 50) -> list[int]:
    """Top-N chunk indices según BM25-smart, del más relevante al menos."""
    query_tokens = tokenize_smart(query)
    scores = bm25_index.get_scores(query_tokens)
    top_indices = np.argsort(scores)[::-1][:top_n]
    return top_indices.tolist()


def rrf_fuse(ranked_lists: list[list[int]], k: int = 60) -> dict[int, float]:
    """Reciprocal Rank Fusion de múltiples rankings."""
    scores = {}
    for ranked_list in ranked_lists:
        for rank, chunk_idx in enumerate(ranked_list):
            scores[chunk_idx] = scores.get(chunk_idx, 0) + 1 / (k + rank)
    return scores


def hybrid_search(query: str, chunks, embeddings_chunks, modelo, bm25_index, top_n: int = 50) -> dict:
    """Hybrid search con RRF. Devuelve top-1 chunk con metadata de ambos sistemas."""
    dense_ranking = get_dense_ranking(query, chunks, embeddings_chunks, modelo, top_n)
    sparse_ranking = get_sparse_ranking(query, chunks, bm25_index, top_n)

    fused = rrf_fuse([dense_ranking, sparse_ranking])
    winner_idx = max(fused, key=fused.get)

    return {
        "chunk_source": chunks[winner_idx]["source"],
        "chunk_text": chunks[winner_idx]["text"],
        "chunk_idx": int(winner_idx),
        "rrf_score": float(fused[winner_idx]),
        "dense_rank": dense_ranking.index(winner_idx) if winner_idx in dense_ranking else None,
        "sparse_rank": sparse_ranking.index(winner_idx) if winner_idx in sparse_ranking else None,
    }


if __name__ == "__main__":
    query = sys.argv[1]

    t0 = time.time()
    chunks = load_chunks(DOCS_DIR)
    embeddings_chunks = load_embeddings_cache(CACHE_PATH, chunks)
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    bm25_index = build_bm25_index(chunks, tokenizer=tokenize_smart)
    t1 = time.time()

    result = hybrid_search(query, chunks, embeddings_chunks, modelo, bm25_index)

    print(f"RRF score: {result['rrf_score']:.6f}")
    print(f"Dense rank: {result['dense_rank']}  |  Sparse rank: {result['sparse_rank']}")
    print(f"\n========== QUERY ==========")
    print(query)
    print(f"\n========== RETRIEVED CHUNK ==========")
    print(f"Archivo: {result['chunk_source']}")
    print(f"Texto (primeros 500 chars):\n{result['chunk_text'][:500]}")
    print(f"\n========== TIEMPOS ==========")
    print(f"Carga sistemas: {t1 - t0:.2f}s")
