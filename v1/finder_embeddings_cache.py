import sys
import time
import hashlib
import numpy as np
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = Path(__file__).parent
CACHE_PATH = SCRIPT_DIR / "embeddings_cache.pkl"
DOCS_DIR = SCRIPT_DIR.parent / "docs-source"

# --- Pieza 1: leer archivos ---
archivos = list(DOCS_DIR.rglob("*.rst")) + list(DOCS_DIR.rglob("*.md"))

# --- Pieza 2: chunking acumulativo (opción C de v0) ---
MIN_PALABRAS = 50
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
        if len(buffer.split()) >= MIN_PALABRAS:
            chunks.append({"source": str(archivo), "text": buffer})
            buffer = ""
    if buffer.strip():
        if chunks and chunks[-1]["source"] == str(archivo):
            chunks[-1]["text"] += "\n\n" + buffer
        else:
            chunks.append({"source": str(archivo), "text": buffer})

print(f"{len(archivos)} archivos, {len(chunks)} chunks")

# --- Pieza 3: hash del corpus para invalidación de caché ---
def corpus_hash(chunks):
    h = hashlib.md5()
    for c in chunks:
        h.update(c["source"].encode())
        h.update(c["text"][:100].encode())
    return h.hexdigest()

current_hash = corpus_hash(chunks)

# --- Pieza 4: embeddings con caché persistente ---
t0 = time.time()

if CACHE_PATH.exists():
    with open(CACHE_PATH, "rb") as f:
        cache = pickle.load(f)
    if cache.get("hash") == current_hash:
        embeddings_chunks = cache["embeddings"]
        print(f"Caché caliente: embeddings cargados desde disco")
    else:
        print(f"Caché invalidado: hash del corpus cambió, recalculando...")
        CACHE_PATH.unlink()
        cache = None

if not CACHE_PATH.exists():
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings_chunks = modelo.encode([c["text"] for c in chunks])
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"hash": current_hash, "embeddings": embeddings_chunks}, f)
    print(f"Caché frío: embeddings calculados y guardados en disco")

t1 = time.time()
print(f"Tiempo de carga/cálculo de embeddings: {t1 - t0:.2f}s")

# --- Pieza 5: query + búsqueda ---
query = sys.argv[1]

t2 = time.time()
t_model_load = time.time()
if "modelo" not in dir():
    modelo = SentenceTransformer("all-MiniLM-L6-v2")
t_model_loaded = time.time()
print(f"Tiempo de carga del modelo: {t_model_loaded - t_model_load:.2f}s")

t_encode = time.time()
embedding_query = modelo.encode([query])
scores = cosine_similarity(embedding_query, embeddings_chunks)
idx = scores.argmax()
t3 = time.time()
t_done = time.time()

print(f"Tiempo de encode query + similarity: {t_done - t_encode:.2f}s")
print(f"Tiempo de query: {t3 - t2:.2f}s")
print(f"\nArchivo: {chunks[idx]['source']}")
print(f"\nTexto:\n{chunks[idx]['text']}")
