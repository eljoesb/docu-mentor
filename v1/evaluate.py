import json
import time
from pathlib import Path
from documentor import (
    load_or_build_chunks,
    load_or_build_embeddings,
    answer_query,
    DOCS_DIR,
    CACHE_PATH,
)

SCRIPT_DIR = Path(__file__).parent
DATASET_PATH = SCRIPT_DIR / "eval_dataset.jsonl"
RESULTS_PATH = SCRIPT_DIR / "eval_results.jsonl"

def load_dataset(path: Path) -> list[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries

def append_result(path: Path, result: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

def main():
    print("=" * 60)
    print("Cargando sistema...")
    print("=" * 60)
    t_start = time.time()
    chunks = load_or_build_chunks(DOCS_DIR)
    embeddings, modelo = load_or_build_embeddings(chunks, CACHE_PATH)
    t_loaded = time.time()
    print(f"Sistema listo en {t_loaded - t_start:.2f}s. {len(chunks)} chunks.")

    dataset = load_dataset(DATASET_PATH)
    print(f"Dataset cargado: {len(dataset)} queries.\n")

    # Limpiar resultados previos
    if RESULTS_PATH.exists():
        RESULTS_PATH.unlink()

    print("=" * 60)
    print("Corriendo evaluación...")
    print("=" * 60)

    for entry in dataset:
        qid = entry["id"]
        query = entry["query"]
        expected = entry["expected"]
        category = entry["category"]

        print(f"\n[{qid:02d}/{len(dataset)}] ({category}, expected={expected})")
        print(f"  Query: {query}")

        t0 = time.time()
        try:
            result = answer_query(query, chunks, embeddings, modelo)
            error = None
        except Exception as e:
            result = None
            error = str(e)
            print(f"  ERROR: {error}")

        t_total = time.time() - t0

        record = {
            "id": qid,
            "query": query,
            "expected": expected,
            "category": category,
            "t_total": round(t_total, 2),
            "error": error,
        }
        if result:
            record.update({
                "chunk_source": result["chunk_source"],
                "chunk_text": result["chunk_text"],
                "chunk_score": round(float(result["chunk_score"]), 4),
                "llm_response": result["llm_response"],
                "t_retrieval": round(result["t_retrieval"], 2),
                "t_llm": round(result["t_llm"], 2),
            })
            print(f"  Score: {record['chunk_score']:.4f} | Total: {t_total:.1f}s")

        append_result(RESULTS_PATH, record)

    print("\n" + "=" * 60)
    print("Evaluación completa.")
    print(f"Resultados guardados en: {RESULTS_PATH}")
    print("=" * 60)

    print_summary_table(dataset, RESULTS_PATH)

def print_summary_table(dataset, results_path):
    results = load_dataset(results_path)
    by_id = {r["id"]: r for r in results}

    print(f"\n{'ID':<4}{'CAT':<18}{'EXP':<8}{'SCORE':<9}{'TIME':<8}QUERY")
    print("-" * 100)
    for entry in dataset:
        r = by_id.get(entry["id"], {})
        score = r.get("chunk_score", "—")
        score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
        time_str = f"{r.get('t_total', 0):.1f}s"
        query_short = entry["query"][:50]
        print(f"{entry['id']:<4}{entry['category']:<18}{entry['expected']:<8}{score_str:<9}{time_str:<8}{query_short}")

if __name__ == "__main__":
    main()
