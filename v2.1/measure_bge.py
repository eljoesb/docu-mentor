"""
measure_bge.py — Retrieval con BAAI/bge-base-en-v1.5 sobre las 21 queries.

Solo retrieval, sin LLM. Produce bge_base_results.jsonl con datos crudos.

Diferencias vs v1/documentor.py:
  - Modelo: BAAI/bge-base-en-v1.5 (768 dim) en vez de all-MiniLM-L6-v2 (384 dim)
  - Prefijo de query: BGE requiere "Represent this sentence for searching
    relevant passages: " prepended a las queries (no a los documentos).
    Sin este prefijo BGE funciona pero pierde ~5-10% de calidad.
  - Caché: embeddings_cache_bge_base.pkl (separado del caché de v1)
  - Sin LLM: solo produce chunk_source, chunk_text, score
"""

import json
import time
import hashlib
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

DOCS_DIR = BASE_DIR / "docs-source"
CACHE_PATH = SCRIPT_DIR / "embeddings_cache_bge_base.pkl"
DATASET_PATH = SCRIPT_DIR / "eval_dataset.jsonl"
RESULTS_PATH = SCRIPT_DIR / "bge_base_results.jsonl"

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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
            print(f"  Caché válido: {cache_path.name}")
            return cache["embeddings"]
        else:
            print(f"  Caché inválido (hash cambió), re-encoding...")
            cache_path.unlink()

    print(f"  Encoding {len(chunks)} chunks (sin prefijo)...")
    t0 = time.time()
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True)
    print(f"  Encoding completado en {time.time() - t0:.1f}s")

    with open(cache_path, "wb") as f:
        pickle.dump({"hash": current_hash, "embeddings": embeddings}, f)
    print(f"  Caché guardado: {cache_path.name}")

    return embeddings


def load_dataset(path):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def main():
    print("=" * 60)
    print(f"measure_bge.py — {MODEL_NAME}")
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
    print("Corriendo retrieval...")
    print(f"{'=' * 60}")

    for entry in dataset:
        qid = entry["id"]
        query = entry["query"]

        t0 = time.time()
        query_embedding = model.encode([QUERY_PREFIX + query])
        scores = cosine_similarity(query_embedding, embeddings)
        idx = int(scores.argmax())
        best_score = float(scores[0][idx])
        t_retrieval = time.time() - t0

        record = {
            "id": qid,
            "query": query,
            "category": entry["category"],
            "expected": entry["expected"],
            "chunk_source": chunks[idx]["source"],
            "chunk_text": chunks[idx]["text"],
            "score": round(best_score, 4),
            "t_retrieval": round(t_retrieval, 3),
        }

        with open(RESULTS_PATH, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"  Q{qid:>2}: score={best_score:.4f}  t={t_retrieval:.3f}s  "
              f"src={Path(chunks[idx]['source']).name}")

    print(f"\n{'=' * 60}")
    print(f"Resultados guardados en: {RESULTS_PATH}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
