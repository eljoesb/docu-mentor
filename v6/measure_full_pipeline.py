"""
measure_full_pipeline.py — Pipeline v6: top-K=3 retrieval con llama3.1:8b.

Unico cambio respecto a v5.1: retrieval top-3 en vez de top-1,
prompt adaptado a contextos multiples numerados.

Produce pipeline_results_v6.jsonl con:
  query, chunks (top-3), llm_response, t_retrieval, t_llm, t_total
"""

import json
import time
import hashlib
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

DOCS_DIR = BASE_DIR / "docs-source"
CACHE_PATH = BASE_DIR / "v3" / "embeddings_cache_bge_base.pkl"
DATASET_PATH = BASE_DIR / "v3" / "eval_dataset.jsonl"
RESULTS_PATH = SCRIPT_DIR / "pipeline_results_v6.jsonl"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"
TOP_K = 3

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about technical documentation.

Use ONLY the following contexts to answer the question. If none of the contexts contain the answer, say "I don't know based on the provided context."

If the contexts discuss related topics but don't directly answer the question, say so explicitly instead of adapting the contexts to fit the question.

{contexts}

Question: {question}

Answer:"""


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
        else:
            print(f"  Cache invalido (hash cambio), re-encoding...")

    print(f"  Encoding {len(chunks)} chunks (sin prefijo)...")
    t0 = time.time()
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)
    print(f"  Encoding completado en {time.time() - t0:.1f}s")

    with open(cache_path, "wb") as f:
        pickle.dump({"hash": current_hash, "embeddings": embeddings}, f)
    print(f"  Cache guardado: {cache_path}")

    return embeddings


def load_dataset(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def generate(prompt, model=OLLAMA_MODEL):
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    return response.json()["response"]


def retrieve_top_k(query, model, chunks, embeddings, k=TOP_K):
    """Top-K retrieval con BGE-base."""
    query_embedding = model.encode([QUERY_PREFIX + query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for idx in top_indices:
        results.append({
            "source": chunks[idx]["source"],
            "text": chunks[idx]["text"],
            "score": float(scores[idx]),
        })
    return results


def format_contexts(chunk_results):
    """Formatea chunks como contextos numerados."""
    parts = []
    for i, chunk in enumerate(chunk_results, 1):
        parts.append(f"Context {i}:\n{chunk['text']}")
    return "\n\n".join(parts)


def main():
    print("=" * 60)
    print(f"v6 measure_full_pipeline.py — top-K={TOP_K}")
    print(f"  Retrieval: {MODEL_NAME}")
    print(f"  Generation: {OLLAMA_MODEL} via Ollama")
    print("=" * 60)

    print(f"\nCargando modelo {MODEL_NAME}...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"  Modelo cargado en {time.time() - t0:.1f}s")

    print("\nPreparando corpus...")
    chunks = load_or_build_chunks(DOCS_DIR)
    print(f"  {len(chunks)} chunks")
    embeddings = load_or_build_embeddings(chunks, model, CACHE_PATH)

    dataset = load_dataset(DATASET_PATH)
    print(f"\nDataset: {len(dataset)} queries")

    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    print(f"\n{'=' * 60}")
    print(f"Corriendo pipeline (top-{TOP_K} retrieval + generation)...")
    print(f"{'=' * 60}")

    for entry in dataset:
        qid = entry["id"]
        query = entry["query"]

        # --- Retrieval top-K ---
        t0 = time.time()
        top_chunks = retrieve_top_k(query, model, chunks, embeddings)
        t_retrieval = time.time() - t0

        # --- Generation ---
        contexts_str = format_contexts(top_chunks)
        prompt = PROMPT_TEMPLATE.format(contexts=contexts_str, question=query)
        t1 = time.time()
        try:
            llm_response = generate(prompt)
            llm_error = None
        except Exception as e:
            llm_response = None
            llm_error = str(e)
        t_llm = time.time() - t1

        t_total = t_retrieval + t_llm

        record = {
            "id": qid,
            "query": query,
            "category": entry["category"],
            "expected": entry["expected"],
            "chunks": [
                {
                    "source": c["source"],
                    "text": c["text"],
                    "score": round(c["score"], 4),
                }
                for c in top_chunks
            ],
            "llm_response": llm_response,
            "llm_error": llm_error,
            "t_retrieval": round(t_retrieval, 3),
            "t_llm": round(t_llm, 3),
            "t_total": round(t_total, 3),
        }

        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        sources = [Path(c["source"]).name for c in top_chunks]
        scores = [f"{c['score']:.3f}" for c in top_chunks]
        status = "OK" if llm_response else f"ERR: {llm_error}"
        print(f"  Q{qid:>2}: scores=[{','.join(scores)}]  "
              f"t_ret={t_retrieval:.2f}s  t_llm={t_llm:.2f}s  "
              f"src=[{','.join(sources)}]  [{status}]")

    print(f"\n{'=' * 60}")
    print(f"Resultados guardados en: {RESULTS_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
