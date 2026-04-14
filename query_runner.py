from __future__ import annotations

import json
import sys
from dataclasses import asdict

from kg_env import load_env_file
from kg_infused_rag import KGInfusedRAG
from neo4j_client import Neo4jClient, Neo4jConfig
from retrieval_wikipedia import wikipedia_search_passages
from spreading_activation import GeminiLLM, OllamaLLM
from llm_language import detect_language, wrap_prompt


def answer_question(
    question: str,
    *,
    llm: str = "ollama",
    domain: str | None = None,
    rounds: int = 3,
    seed_k: int = 5,
    entities_per_round: int = 10,
    wiki_lang: str = "tr",
    passage_k: int = 8,
    mode: str = "kg_rag",
) -> dict:
    """
    Single entry function:
      - mode=no_retrieval: LLM only
      - mode=vanilla_rag: Wikipedia snippets only
      - mode=kg_rag: KG-Infused RAG (Modules 1-3)
    """
    load_env_file(".env")
    llm_client = OllamaLLM() if llm == "ollama" else GeminiLLM()
    lang = detect_language(question)

    if mode == "no_retrieval":
        ans = llm_client.generate(
            wrap_prompt(f"Answer the question.\nQuestion: {question}\nAnswer:", lang=lang)
        ).strip()
        return {"mode": mode, "question": question, "answer": ans}

    if mode == "vanilla_rag":
        passages = wikipedia_search_passages(question, k=passage_k, lang=wiki_lang)
        bullets = "\n".join([f"- {p.title}: {p.snippet}" for p in passages[:12]])
        note = llm_client.generate(
            wrap_prompt(
                "Given a question and Wikipedia snippets, write a compact note with only useful info.\n"
                f"Question: {question}\nSnippets:\n{bullets}\nNote:",
                lang=lang,
            )
        ).strip()
        ans = llm_client.generate(
            wrap_prompt(
                "Answer the question using the note. If insufficient, say what is missing.\n"
                f"Question: {question}\nNote: {note}\nAnswer:",
                lang=lang,
            )
        ).strip()
        return {
            "mode": mode,
            "question": question,
            "passages": [p.__dict__ for p in passages],
            "passage_note": note,
            "answer": ans,
        }

    if mode != "kg_rag":
        raise ValueError("mode must be one of: no_retrieval, vanilla_rag, kg_rag")

    cfg = Neo4jConfig.from_env()
    neo = Neo4jClient(cfg)
    try:
        neo.verify()
        rag = KGInfusedRAG(
            neo,
            llm=llm_client,
            domain=domain,
            seed_k=seed_k,
            max_rounds=rounds,
            entities_per_round=entities_per_round,
            wiki_lang=wiki_lang,
            passage_k=passage_k,
        )
        res = rag.answer(question)
        return asdict(res)
    finally:
        neo.close()


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python query_runner.py \"your question\" [kg_rag|vanilla_rag|no_retrieval]")
    q = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) >= 3 else "kg_rag"
    out = answer_question(q, mode=mode)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

