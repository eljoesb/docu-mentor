# DocuMentor

Asistente de consulta sobre documentación técnica. Proyecto guía del roadmap de transición a AI/ML Engineer.

## Versiones

- **[v0](./v0/README.md)** — Retrieval léxico y semántico. Recorre los primeros 30 años de Information Retrieval en una sola sesión: grep tonto → TF-IDF → chunking → embeddings.
- **[v1](./v1/README.md)** — Pipeline RAG completo (caché persistente + generación con Ollama) + evaluación sistemática de 21 queries. El sistema acierta 48% del tiempo por las razones equivocadas; el eval lo demuestra.
- **v2** _(en progreso)_ — Ataque al cuello de botella estructural identificado en v1: retrieval.

## Corpus actual

Documentación oficial de PyTorch (`docs/source`, 248 archivos).

## Stack v1

Python + scikit-learn + sentence-transformers + Ollama (llama3.2:3b local).
