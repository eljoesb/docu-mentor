import sys
from pathlib import Path

# Pieza 1: leer archivos
archivos = list(Path("../docs-source").rglob("*.rst")) + list(Path("../docs-source").rglob("*.md"))

# Pieza 2: chunking acumulativo (opción C)
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

longitudes = [len(c["text"].split()) for c in chunks]
print(f"{len(archivos)} archivos, {len(chunks)} chunks")
print(f"min: {min(longitudes)}, max: {max(longitudes)}")
print(f"< 50: {sum(1 for l in longitudes if l < 50)}")
print(f"< 100: {sum(1 for l in longitudes if l < 100)}")
print(f"< 200: {sum(1 for l in longitudes if l < 200)}")
print(f"< 500: {sum(1 for l in longitudes if l < 500)}")
print(f">= 500: {sum(1 for l in longitudes if l >= 500)}")

chunks_ordenados = sorted(chunks, key=lambda c: len(c["text"].split()))
print("\n--- 5 chunks más cortos ---")
for c in chunks_ordenados[:5]:
    print(f"[{len(c['text'].split())} palabras] {c['source']}")
    print(c['text'])
    print("---")

# Pieza 3: query
query = sys.argv[1]

# Pieza 4: ranking con embeddings
import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

modelo = SentenceTransformer("all-MiniLM-L6-v2")

t0 = time.time()
embeddings_chunks = modelo.encode([c["text"] for c in chunks])
embedding_query = modelo.encode([query])
scores = cosine_similarity(embedding_query, embeddings_chunks)
idx = scores.argmax()
t1 = time.time()

print(f"\nTiempo de encoding + búsqueda: {t1 - t0:.1f}s")
print(f"\nArchivo: {chunks[idx]['source']}")
print(f"\nTexto:\n{chunks[idx]['text']}")

