"""
judge.py — LLM-as-judge con GPT-4o-mini sobre los datos de v3b.

Tres pasadas independientes (una por eje), cada una con su propio prompt
que muestra solo la informacion relevante. 63 llamadas totales.

Uso:
    export OPENAI_API_KEY=sk-... 
    python v4/judge.py

Output: judge_retrieval.csv, judge_generation.csv, judge_utility.csv
"""

import csv
import json
import sys
import time
from pathlib import Path
from openai import OpenAI

SCRIPT_DIR = Path(__file__).parent
RESULTS_PATH = SCRIPT_DIR.parent / "v3" / "pipeline_results_v3b.jsonl"

MODEL = "gpt-4o-mini"

PROMPTS = {
    "retrieval": {
        "valid": {"b", "t", "m"},
        "template": """You are evaluating a RAG system's retrieval quality.

Given a user question and the retrieved chunk, classify the retrieval:
- b (good): the chunk directly answers the question
- t (tangential): the chunk discusses a related topic but doesn't directly answer
- m (bad): the chunk is irrelevant

Be strict: if the chunk mentions the topic but doesn't actually answer what was asked, that's tangential, not good.

Important: If the chunk contains terms or words from the question but in a semantically different context (e.g., "detach" used for memory management vs "detach" for preventing overfitting), classify as m (bad), not t (tangential). Surface word overlap without semantic match is bad retrieval.

Question: {query}

Retrieved chunk:
{chunk_text}

Respond with exactly one letter: b, t, or m.""",
    },
    "generation": {
        "valid": {"f", "x", "i"},
        "template": """You are evaluating whether a language model's response was faithful to the provided context.

Given the context chunk and the model's response, classify the generation:
- f (faithful): the response only uses information from the chunk
- x (mixed): the response mixes chunk information with external knowledge
- i (invented): the response contains claims not supported by the chunk

Note: A response that says "I don't know based on the provided context" is faithful (f), not mixed or invented. Honest refusal counts as faithful.

Context chunk:
{chunk_text}

Model response:
{llm_response}

Respond with exactly one letter: f, x, or i.""",
    },
    "utility": {
        "valid": {"u", "p", "d"},
        "template": """You are evaluating whether a RAG system's response would be useful to a developer who asked the question.

Given only the question and the response (you do NOT see the source chunk), classify utility:
- u (useful): a developer would get what they need from this response
- p (partial): partially helpful but incomplete or not quite what was asked
- d (harmful): incorrect, misleading, or could cause the developer to make mistakes

Important rules:
- If the question is clearly out-of-domain (e.g., asking about cooking, geography, or unrelated topics) and the response honestly says "I don't know", classify as u (useful). Honest refusal of OOD questions is the correct behavior.
- If the response contains specific technical claims that sound plausible but are factually incorrect, classify as d (harmful), even if the response seems helpful.

Examples:
- Question: "what is the capital of France" / Response: "I don't know based on context" → u (useful, correct OOD rejection)
- Question: "difference between .size() and .shape" / Response: invents a difference that doesn't exist in PyTorch → d (harmful, confidently wrong)

Question: {query}

System response:
{llm_response}

Respond with exactly one letter: u, p, or d.""",
    },
}


def load_results(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r["id"])
    return records


def short_source(path_str):
    if not path_str:
        return "?"
    p = Path(path_str)
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "docs-source" and i + 1 < len(parts):
            return "/".join(parts[i + 1:])
    return p.name


def judge_one(client, axis, record):
    """Clasifica una query en un eje. Retorna la etiqueta o 'error'."""
    config = PROMPTS[axis]
    prompt = config["template"].format(
        query=record.get("query", ""),
        chunk_text=record.get("chunk_text", ""),
        llm_response=record.get("llm_response", ""),
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1,
    )

    label = response.choices[0].message.content.strip().lower()

    if label not in config["valid"]:
        print(f"    WARNING: respuesta inesperada '{label}', marcando como error")
        return "error"

    return label


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query", "source", "label"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_axis(client, axis, records):
    """Corre el juez para un eje completo. Retorna lista de rows."""
    print(f"\n{'=' * 50}")
    print(f"  Eje: {axis.upper()}")
    print(f"{'=' * 50}")

    rows = []
    for rec in records:
        qid = rec["id"]
        query = rec["query"]
        source = short_source(rec.get("chunk_source", ""))

        label = judge_one(client, axis, rec)

        rows.append({
            "query_id": qid,
            "query": query,
            "source": source,
            "label": label,
        })

        print(f"  Q{qid:>2}: {label}  ({source})")

    return rows


def main():
    client = OpenAI()

    if not RESULTS_PATH.exists():
        print(f"Error: {RESULTS_PATH} no existe", file=sys.stderr)
        sys.exit(1)

    records = load_results(RESULTS_PATH)
    print(f"Cargados {len(records)} records de {RESULTS_PATH.name}")
    print(f"Modelo juez: {MODEL}, temperature=0, max_tokens=1")

    for axis in ["retrieval", "generation", "utility"]:
        t0 = time.time()
        rows = run_axis(client, axis, records)
        elapsed = time.time() - t0

        csv_path = SCRIPT_DIR / f"judge_{axis}.csv"
        write_csv(csv_path, rows)

        labels = [r["label"] for r in rows]
        errors = labels.count("error")
        print(f"\n  Guardado: {csv_path.name}  ({elapsed:.1f}s, {errors} errores)")

    print(f"\n{'=' * 50}")
    print("Juez completo. Corre compare.py para ver acuerdo.")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
