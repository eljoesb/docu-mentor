import re
import sys
import time
from pathlib import Path
from rank_bm25 import BM25Okapi

SCRIPT_DIR = Path(__file__).parent
DOCS_DIR = SCRIPT_DIR.parent / "docs-source"


def load_chunks(docs_dir: Path, min_palabras: int = 50) -> list[dict]:
    """Lee los archivos y hace el chunking. Devuelve lista de {source, text}.
    Copiado textual de v1/documentor.py para mantener el mismo corpus."""
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


def tokenize(text: str) -> list[str]:
    """Tokenización mínima: lowercase + split por espacios."""
    return text.lower().split()


def tokenize_smart(text: str) -> list[str]:
    """Tokenización que rompe por puntos, guiones, paréntesis y otros separadores
    comunes en nombres de API, además de espacios y lowercase."""
    text = text.lower()
    text = re.sub(r"[.\-_()/\\\[\]{}'\"`]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens


def build_bm25_index(chunks: list[dict], tokenizer=tokenize) -> BM25Okapi:
    """Construye el índice BM25 sobre los chunks tokenizados."""
    corpus = [tokenizer(c["text"]) for c in chunks]
    return BM25Okapi(corpus)


def search(query: str, chunks: list[dict], index: BM25Okapi, tokenizer=tokenize) -> dict:
    """Busca la query en el índice BM25. Devuelve el top-1 chunk con score."""
    query_tokens = tokenizer(query)
    scores = index.get_scores(query_tokens)
    idx = scores.argmax()
    return {
        "chunk_source": chunks[idx]["source"],
        "chunk_text": chunks[idx]["text"],
        "chunk_score": float(scores[idx]),
        "chunk_idx": int(idx),
    }


if __name__ == "__main__":
    query = sys.argv[1]

    t0 = time.time()
    chunks = load_chunks(DOCS_DIR)
    index = build_bm25_index(chunks)
    t1 = time.time()

    result = search(query, chunks, index)

    print(f"BM25 score: {result['chunk_score']:.4f}")
    print(f"Chunks en el corpus: {len(chunks)}")
    print(f"\n========== QUERY ==========")
    print(query)
    print(f"\n========== RETRIEVED CHUNK ==========")
    print(f"Archivo: {result['chunk_source']}")
    print(f"Texto (primeros 500 chars):\n{result['chunk_text'][:500]}")
    print(f"\n========== TIEMPOS ==========")
    print(f"Carga + indexación: {t1 - t0:.2f}s")
