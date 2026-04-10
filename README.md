# DocuMentor

Asistente de consulta sobre documentación técnica. Proyecto guía del roadmap de transición a AI/ML Engineer.

## Versiones

- **[v0](./v0/README.md)** — Retrieval léxico y semántico. Recorre los primeros 30 años de Information Retrieval en una sola sesión: grep tonto → TF-IDF → chunking → embeddings.
- **[v1](./v1/README.md)** — Pipeline RAG completo (caché persistente + generación con Ollama) + evaluación sistemática de 21 queries. El sistema acierta 48% del tiempo por las razones equivocadas; el eval lo demuestra.
- **[v2](./v2/README.md)** — Hybrid search con BM25 + RRF para atacar el cuello de botella de retrieval de v1. Hipótesis: combinar dense (embeddings) + sparse (BM25) vía Reciprocal Rank Fusion recupera más chunks relevantes. Resultado: 2/21 buenos, peor que embeddings puros. El diagnóstico reveló que el problema no es el algoritmo de retrieval sino la estructura del corpus indexado.
- **[v2.1](./v2.1/README.md)** — Cambio de modelo de embeddings: MiniLM-L6-v2 (384 dim) → BGE-base-en-v1.5 (768 dim). Resultado: 11/21 buenos, casi el triple del baseline. El modelo de embeddings era el cuello de botella. Camino 3 confirmado.

## Estado actual

**Modelo de embeddings activo:** BAAI/bge-base-en-v1.5 (768 dim, prefix asimétrico para queries).

**Retrieval:** 11/21 buenos bajo criterio estricto. 5 queries OOD/opinión (irresolubles por diseño), 5 fallos reales del sistema.

**Hipótesis validada:** el cuello de botella era capacidad semántica del modelo, no el algoritmo de búsqueda ni la estructura del corpus. Pasar de MiniLM a BGE-base fue suficiente para cruzar el umbral de 10/21.

**Hipótesis abiertas:**
- Hybrid search (BGE + BM25-smart) podría recuperar las regresiones de retrieval puramente denso (Q13: BGE pierde especificidad de keywords que BM25 capturaría). v2 hybrid falló porque el componente denso era débil; con BGE como base, el balance de RRF cambia.
- Chunking más granular podría mejorar queries donde el chunk relevante existe pero está diluido en un bloque de 50+ palabras.
- BGE-large (1024 dim) queda descartado como paso inmediato — BGE-base ya cruzó el umbral. Si hybrid con BGE-base no alcanza, escalar modelo es una opción de reserva.

## Corpus actual

Documentación oficial de PyTorch (`docs/source`, 248 archivos).

## Stack v2.1

Python + scikit-learn + sentence-transformers (BAAI/bge-base-en-v1.5) + rank_bm25 + Ollama (llama3.2:3b local).
