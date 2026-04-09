import sys
from pathlib import Path

# Pieza 1: leer archivos
archivos = list(Path("../docs-source").rglob("*.rst")) + list(Path("../docs-source").rglob("*.md"))

# Pieza 2: chunking por párrafos
chunks = []
for archivo in archivos:
    texto = archivo.read_text(encoding="utf-8", errors="ignore")
    parrafos = texto.split("\n\n")
    for parrafo in parrafos:
        parrafo = parrafo.strip()
        if parrafo:
            chunks.append({"source": str(archivo), "text": parrafo})

print(f"{len(archivos)} archivos, {len(chunks)} chunks")

# Pieza 3: query
query = sys.argv[1]

# Pieza 4: ranking tonto
palabras_query = query.lower().split()
mejor_chunk = max(chunks, key=lambda c: sum(1 for p in palabras_query if p in c["text"].lower()))

print(f"\nArchivo: {mejor_chunk['source']}")
print(f"\nTexto:\n{mejor_chunk['text']}")
