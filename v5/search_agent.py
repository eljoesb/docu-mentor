"""
search_agent.py — v5: Re-busqueda con query reformulada.

Reutiliza retrieval (BGE-base) y generation (Ollama llama3.2:3b) de v3.
Agrega: deteccion de respuesta insuficiente + reformulacion + reintento.

Produce pipeline_results_v5.jsonl con campos para analisis:
  query, original_chunk, original_chunk_score, original_response,
  retried, reformulated_query, retry_chunk, retry_chunk_score, final_response
"""

import json
import time
import hashlib
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

DOCS_DIR = BASE_DIR / "docs-source"
CACHE_PATH = BASE_DIR / "v3" / "embeddings_cache_bge_base.pkl"
DATASET_PATH = BASE_DIR / "v3" / "eval_dataset.jsonl"
RESULTS_PATH = SCRIPT_DIR / "pipeline_results_v5.jsonl"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# --- Prompt de generation: identico a v3b ---
PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about technical documentation.

Use ONLY the following context to answer the question. If the context doesn't contain the answer, say "I don't know based on the provided context."

If the context discusses a related topic but doesn't directly answer the question, say so explicitly instead of adapting the context to fit the question.

Context:
{context}

Question: {question}

Answer:"""

# --- Prompt de reformulacion: el componente nuevo de v5 ---
REFORMULATE_TEMPLATE = """A PyTorch documentation search failed to answer a question. The retrieved passage was not relevant enough.

Original question: {query}

Retrieved passage (not useful):
{chunk}

System response: {response}

Write a single search query that would find the correct PyTorch documentation to answer the original question. Use specific PyTorch class names, function names, or technical terms instead of general language. Do not explain, just write the query.

Query:"""


# ── Corpus / embeddings (reutilizado de v3) ─────────────────────────────

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
    return embeddings


def load_dataset(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ── Componentes del pipeline ─────────────────────────────────────────────

def retrieve(query, model, chunks, embeddings):
    """Top-1 retrieval con BGE-base. Identico a v3."""
    query_embedding = model.encode([QUERY_PREFIX + query])
    scores = cosine_similarity(query_embedding, embeddings)
    idx = int(scores.argmax())
    return {
        "chunk_source": chunks[idx]["source"],
        "chunk_text": chunks[idx]["text"],
        "score": float(scores[0][idx]),
    }


def generate(query, chunk_text):
    """Generation con Ollama llama3.2:3b. Prompt identico a v3b."""
    prompt = PROMPT_TEMPLATE.format(context=chunk_text, question=query)
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    return response.json()["response"]


def needs_retry(response):
    """Detector de respuesta insuficiente. Heuristicas literales, sin inventar."""
    r = response.lower()
    return (
        "i don't know" in r
        or "the context doesn't" in r
        or "the context does not" in r
    )


def reformulate(original_query, bad_chunk, bad_response):
    """Reformula la query usando Ollama. El componente nuevo de v5."""
    chunk_truncated = bad_chunk[:500]
    prompt = REFORMULATE_TEMPLATE.format(
        query=original_query,
        chunk=chunk_truncated,
        response=bad_response[:300],
    )
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    new_query = response.json()["response"].strip()
    # Limpiar: si el modelo agrega comillas o prefijos, sacarlos
    new_query = new_query.strip('"').strip("'")
    if new_query.lower().startswith("query:"):
        new_query = new_query[6:].strip()
    return new_query


# ── Orquestador ──────────────────────────────────────────────────────────

def run(entry, model, chunks, embeddings):
    """Orquesta: retrieve → generate → si needs_retry → reformulate → retrieve → generate.
    Maximo 1 reintento. Devuelve dict con todos los campos para analisis."""
    qid = entry["id"]
    query = entry["query"]

    # --- Intento original ---
    t0 = time.time()
    r1 = retrieve(query, model, chunks, embeddings)
    t_retrieval_1 = time.time() - t0

    t1 = time.time()
    try:
        original_response = generate(query, r1["chunk_text"])
        original_error = None
    except Exception as e:
        original_response = f"ERROR: {e}"
        original_error = str(e)
    t_gen_1 = time.time() - t1

    # --- Decidir si reintentar ---
    retried = False
    reformulated_query = None
    retry_chunk_source = None
    retry_chunk_text = None
    retry_chunk_score = None
    retry_response = None
    t_reformulate = 0
    t_retrieval_2 = 0
    t_gen_2 = 0

    if original_error is None and needs_retry(original_response):
        retried = True

        # Reformular
        t2 = time.time()
        try:
            reformulated_query = reformulate(query, r1["chunk_text"], original_response)
        except Exception as e:
            reformulated_query = f"ERROR: {e}"
        t_reformulate = time.time() - t2

        # Segundo retrieval con query reformulada
        if reformulated_query and not reformulated_query.startswith("ERROR"):
            t3 = time.time()
            r2 = retrieve(reformulated_query, model, chunks, embeddings)
            t_retrieval_2 = time.time() - t3

            retry_chunk_source = r2["chunk_source"]
            retry_chunk_text = r2["chunk_text"]
            retry_chunk_score = round(r2["score"], 4)

            # Segunda generation con query ORIGINAL + chunk nuevo
            t4 = time.time()
            try:
                retry_response = generate(query, r2["chunk_text"])
            except Exception as e:
                retry_response = f"ERROR: {e}"
            t_gen_2 = time.time() - t4

    # --- Final response ---
    final_response = retry_response if retried and retry_response else original_response

    return {
        "id": qid,
        "query": query,
        "category": entry["category"],
        "expected": entry["expected"],
        # Intento original
        "original_chunk_source": r1["chunk_source"],
        "original_chunk": r1["chunk_text"],
        "original_chunk_score": round(r1["score"], 4),
        "original_response": original_response,
        # Reintento
        "retried": retried,
        "reformulated_query": reformulated_query,
        "retry_chunk_source": retry_chunk_source,
        "retry_chunk": retry_chunk_text,
        "retry_chunk_score": retry_chunk_score,
        "retry_response": retry_response,
        # Resultado final
        "final_response": final_response,
        # Timing
        "t_retrieval_1": round(t_retrieval_1, 3),
        "t_gen_1": round(t_gen_1, 3),
        "t_reformulate": round(t_reformulate, 3),
        "t_retrieval_2": round(t_retrieval_2, 3),
        "t_gen_2": round(t_gen_2, 3),
        "t_total": round(t_retrieval_1 + t_gen_1 + t_reformulate + t_retrieval_2 + t_gen_2, 3),
    }


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("v5 search_agent.py — Re-busqueda con query reformulada")
    print(f"  Retrieval: {MODEL_NAME}")
    print(f"  Generation + Reformulation: {OLLAMA_MODEL} via Ollama")
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
    print("Corriendo pipeline v5 (retrieve → generate → retry?)...")
    print(f"{'=' * 60}\n")

    for entry in dataset:
        result = run(entry, model, chunks, embeddings)

        # Escritura incremental — no perder datos por crash
        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        retry_str = ""
        if result["retried"]:
            retry_str = f" → RETRY (reformulated: {result['reformulated_query'][:60]}...)"

        print(f"  Q{result['id']:>2}: score={result['original_chunk_score']:.4f}"
              f"  t={result['t_total']:.1f}s"
              f"  src={Path(result['original_chunk_source']).name}"
              f"{retry_str}")

    print(f"\n{'=' * 60}")
    print(f"Resultados guardados en: {RESULTS_PATH}")

    # Resumen
    results = load_dataset(RESULTS_PATH)
    n_retried = sum(1 for r in results if r["retried"])
    print(f"  Total queries: {len(results)}")
    print(f"  Retried: {n_retried}")
    print(f"  No retry: {len(results) - n_retried}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
