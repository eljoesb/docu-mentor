import sys
import time
import hashlib
import numpy as np
import pickle
import requests
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = Path(__file__).parent
CACHE_PATH = SCRIPT_DIR / "embeddings_cache.pkl"
DOCS_DIR = SCRIPT_DIR.parent / "docs-source"

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about PyTorch documentation.

Use ONLY the following context to answer the question. If the context doesn't contain the answer, say "I don't know based on the provided context."

Context:
{context}

Question: {question}

Answer:"""


def load_or_build_chunks(docs_dir: Path, min_palabras: int = 50) -> list[dict]:
    """Lee los archivos y hace el chunking. Devuelve lista de {source, text}."""
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


def corpus_hash(chunks: list[dict]) -> str:
    h = hashlib.md5()
    for c in chunks:
        h.update(c["source"].encode())
        h.update(c["text"][:100].encode())
    return h.hexdigest()


def load_or_build_embeddings(chunks: list[dict], cache_path: Path):
    """Carga embeddings del caché o los recalcula. Devuelve (embeddings, modelo)."""
    current_hash = corpus_hash(chunks)
    embeddings_chunks = None

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if cache.get("hash") == current_hash:
            embeddings_chunks = cache["embeddings"]
        else:
            cache_path.unlink()

    modelo = SentenceTransformer("all-MiniLM-L6-v2")

    if embeddings_chunks is None:
        embeddings_chunks = modelo.encode([c["text"] for c in chunks])
        with open(cache_path, "wb") as f:
            pickle.dump({"hash": current_hash, "embeddings": embeddings_chunks}, f)

    return embeddings_chunks, modelo


def answer_query(query: str, chunks, embeddings, modelo, ollama_model: str = "llama3.2:3b"):
    """Hace retrieval + generación para una query. Devuelve dict con todos los datos."""
    t0 = time.time()
    embedding_query = modelo.encode([query])
    scores = cosine_similarity(embedding_query, embeddings)
    idx = scores.argmax()
    best_score = scores[0][idx]
    t1 = time.time()

    prompt = PROMPT_TEMPLATE.format(context=chunks[idx]["text"], question=query)

    t2 = time.time()
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": ollama_model,
            "prompt": prompt,
            "stream": False
        }
    )
    respuesta = response.json()["response"]
    t3 = time.time()

    return {
        "chunk_source": chunks[idx]["source"],
        "chunk_text": chunks[idx]["text"],
        "chunk_score": best_score,
        "llm_response": respuesta,
        "t_retrieval": t1 - t0,
        "t_llm": t3 - t2,
    }


if __name__ == "__main__":
    query = sys.argv[1]

    t_start = time.time()
    chunks = load_or_build_chunks(DOCS_DIR)
    embeddings, modelo = load_or_build_embeddings(chunks, CACHE_PATH)
    t_loaded = time.time()

    result = answer_query(query, chunks, embeddings, modelo)

    print(f"Best chunk score: {result['chunk_score']:.4f}")
    print(f"\n========== QUERY ==========")
    print(query)
    print(f"\n========== RETRIEVED CHUNK ==========")
    print(f"Archivo: {result['chunk_source']}")
    print(f"Texto:\n{result['chunk_text']}")
    print(f"\n========== LLM RESPONSE ==========")
    print(result["llm_response"])
    print(f"\n========== TIEMPOS ==========")
    print(f"Carga sistema:      {t_loaded - t_start:.2f}s")
    print(f"Retrieval:           {result['t_retrieval']:.2f}s")
    print(f"LLM (Ollama):        {result['t_llm']:.2f}s")
