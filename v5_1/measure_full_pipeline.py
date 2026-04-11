"""
measure_full_pipeline.py — Pipeline v5.1: identico a v3b con llama3.1:8b.

Unico cambio respecto a v3/measure_full_pipeline.py: OLLAMA_MODEL.
Retrieval con BAAI/bge-base-en-v1.5, generation con Ollama llama3.1:8b.

Produce pipeline_results_v5_1.jsonl con:
  query, chunk_source, chunk_text, score, llm_response,
  t_retrieval, t_llm, t_total
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
RESULTS_PATH = SCRIPT_DIR / "pipeline_results_v5_1.jsonl"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434/api/generate"

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about technical documentation.

Use ONLY the following context to answer the question. If the context doesn't contain the answer, say "I don't know based on the provided context."

If the context discusses a related topic but doesn't directly answer the question, say so explicitly instead of adapting the context to fit the question.

Context:
{context}

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


def main():
    print("=" * 60)
    print(f"v5.1 measure_full_pipeline.py")
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
    print("Corriendo pipeline completo (retrieval + generation)...")
    print(f"{'=' * 60}")

    for entry in dataset:
        qid = entry["id"]
        query = entry["query"]

        # --- Retrieval ---
        t0 = time.time()
        query_embedding = model.encode([QUERY_PREFIX + query])
        scores = cosine_similarity(query_embedding, embeddings)
        idx = int(scores.argmax())
        best_score = float(scores[0][idx])
        t_retrieval = time.time() - t0

        chunk_source = chunks[idx]["source"]
        chunk_text = chunks[idx]["text"]

        # --- Generation ---
        prompt = PROMPT_TEMPLATE.format(context=chunk_text, question=query)
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
            "chunk_source": chunk_source,
            "chunk_text": chunk_text,
            "score": round(best_score, 4),
            "llm_response": llm_response,
            "llm_error": llm_error,
            "t_retrieval": round(t_retrieval, 3),
            "t_llm": round(t_llm, 3),
            "t_total": round(t_total, 3),
        }

        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        status = "OK" if llm_response else f"ERR: {llm_error}"
        print(f"  Q{qid:>2}: score={best_score:.4f}  "
              f"t_ret={t_retrieval:.2f}s  t_llm={t_llm:.2f}s  "
              f"src={Path(chunk_source).name}  [{status}]")

    print(f"\n{'=' * 60}")
    print(f"Resultados guardados en: {RESULTS_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
