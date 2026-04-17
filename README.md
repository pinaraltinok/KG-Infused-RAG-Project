# SocialRAG (KG‑Infused RAG – Türkiye / Wikidata5M)

This repo implements the course project pipeline:

- **No Retrieval** (LLM only)
- **Vanilla RAG** (Wikipedia snippets)
- **Vanilla QE** (LLM query expansion + dual retrieval, no KG)
- **KG‑Infused RAG** (Neo4j + spreading activation + LLM triple selection + KG‑based QE + KG‑augmented generation)

## Setup

1. Ensure Neo4j is running and your `.env` contains at least:

```
NEO4J_PASSWORD=...
```

Optional:

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
```

2. Install Python deps:

```bash
pip install -r requirements.txt
```

## Run API (local)

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

- UI: `http://127.0.0.1:8000/`
- API:
  - `POST /api/ask` returns JSON
  - `POST /api/answer` returns text (answer + short explanation)

## Evaluate on the 50‑question dataset

Your question file should be JSON with either:

- a list of objects, or
- `{ "questions": [ ... ] }`

Each question object must have at least:

- `question_id`
- `question_text`
- `gold_answer`

Run:

```bash
python evaluation.py --questions path\\to\\turkiye_qa.json --mode kg_rag --llm ollama --wiki-lang tr
python evaluation.py --questions path\\to\\turkiye_qa.json --mode vanilla_qe
python evaluation.py --questions path\\to\\turkiye_qa.json --mode vanilla_rag
python evaluation.py --questions path\\to\\turkiye_qa.json --mode no_retrieval
```

