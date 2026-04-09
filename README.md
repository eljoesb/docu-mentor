# DocuMentor

Asistente de consulta sobre documentación técnica. Proyecto guía del roadmap de transición a AI/ML Engineer.

## Versiones

- **[v0](./v0/README.md)** — Retrieval léxico y semántico. Recorre los primeros 30 años de Information Retrieval en una sola sesión: grep tonto → TF-IDF → chunking → embeddings.
- **[v1](./v1/README.md)** — Pipeline RAG completo (caché persistente + generación con Ollama) + evaluación sistemática de 21 queries. El sistema acierta 48% del tiempo por las razones equivocadas; el eval lo demuestra.
- **[v2](./v2/README.md)** — Hybrid search con BM25 + RRF para atacar el cuello de botella de retrieval de v1. Hipótesis: combinar dense (embeddings) + sparse (BM25) vía Reciprocal Rank Fusion recupera más chunks relevantes. Resultado: 2/21 buenos, peor que embeddings puros. El diagnóstico reveló que el problema no es el algoritmo de retrieval sino la estructura del corpus indexado.

## Corpus actual

Documentación oficial de PyTorch (`docs/source`, 248 archivos).

## Stack v2

Python + scikit-learn + sentence-transformers + rank_bm25 + Ollama (llama3.2:3b local).
