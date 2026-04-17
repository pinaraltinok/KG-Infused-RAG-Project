from __future__ import annotations

import json
import sys
from dataclasses import asdict

from kg_env import load_env_file
from kg_infused_rag import KGInfusedRAG
from kg_path_answer import classify_question, kg_path_answer, format_answer_for_lang
from neo4j_client import Neo4jClient, Neo4jConfig
from retrieval_wikipedia import wikipedia_search_passages
from spreading_activation import GeminiLLM, OllamaLLM
from llm_language import detect_language, wrap_prompt


def _auto_domain(question: str) -> str | None:
    """Cheap domain inference from keywords so the user doesn't have to
    pass a `domain=` parameter explicitly."""
    q = (question or "").lower()
    if any(w in q for w in (
        "film", "filmin", "filmler", "yönetmen", "yonetmen", "oynayan",
        "movie", "director", "cinema", "aktör", "aktor",
    )):
        return "cinema"
    if any(w in q for w in (
        "futbol", "takım", "takim", "kulüb", "kulub", "club", "coach",
        "teknik direktör", "teknik direktor", "stadium", "stadyum",
    )):
        return "football"
    if any(w in q for w in (
        "şirket", "sirket", "banka", "holding", "havayol", "airline", "company",
    )):
        return "company"
    if any(w in q for w in (
        "üniversite", "universite", "akadem", "professor", "öğretim",
    )):
        return "academia"
    if any(w in q for w in (
        "albüm", "albüm", "şarkı", "sarkı", "grup", "song", "album", "singer",
    )):
        return "music"
    return None


def _llm_client(llm: str):
    if llm == "ollama":
        return OllamaLLM()
    if llm == "gemini":
        return GeminiLLM()
    raise ValueError(f"Unknown llm: {llm}")


def _safe_generate(llm_client, prompt: str, *, lang: str, style: str, fallback: str = "") -> str:
    """Call the LLM but never raise. Baselines must still return something
    if Ollama or Gemini is misconfigured."""
    try:
        return llm_client.generate(wrap_prompt(prompt, lang=lang, style=style)).strip()
    except Exception as exc:  # noqa: BLE001
        return fallback or f"[llm_unavailable: {type(exc).__name__}]"


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
      - mode=vanilla_rag:  Wikipedia snippets only
      - mode=vanilla_qe:   LLM query expansion (no KG), dual retrieval
      - mode=kg_rag:       KG-Infused RAG (Modules 1-3) — deterministic path
                           solver when the question matches a verified template,
                           otherwise full LLM-guided pipeline.
    """
    load_env_file(".env")
    lang = detect_language(question)
    domain = domain or _auto_domain(question)

    # NoR baseline — pure LLM parametric knowledge.
    if mode == "no_retrieval":
        llm_client = _llm_client(llm)
        ans = _safe_generate(
            llm_client,
            f"Answer the question.\nQuestion: {question}\nAnswer:",
            lang=lang,
            style="final",
            fallback=("Bilinmiyor" if lang == "tr" else "Unknown"),
        )
        return {"mode": mode, "question": question, "answer": ans}

    # Vanilla RAG baseline — Wikipedia snippets only.
    if mode == "vanilla_rag":
        llm_client = _llm_client(llm)
        passages = wikipedia_search_passages(question, k=passage_k, lang=wiki_lang)
        bullets = "\n".join([f"- {p.title}: {p.snippet}" for p in passages[:12]])
        note = _safe_generate(
            llm_client,
            "Given a question and Wikipedia snippets, write a compact note with only useful info.\n"
            f"Question: {question}\nSnippets:\n{bullets}\nNote:",
            lang=lang, style="work", fallback=bullets,
        )
        ans = _safe_generate(
            llm_client,
            "Answer the question using the note. Give the shortest possible final answer — "
            "ideally one noun phrase — plus one brief sentence.\n"
            f"Question: {question}\nNote: {note}\nAnswer:",
            lang=lang, style="final",
            fallback=("Bilinmiyor" if lang == "tr" else "Unknown"),
        )
        return {
            "mode": mode,
            "question": question,
            "passages": [p.__dict__ for p in passages],
            "passage_note": note,
            "answer": ans,
        }

    # Vanilla QE baseline — LLM query expansion without KG, dual retrieval.
    if mode == "vanilla_qe":
        llm_client = _llm_client(llm)
        expanded_query = _safe_generate(
            llm_client,
            "Generate a new short query that is distinct from but related to the original. "
            "Use it to retrieve additional relevant Wikipedia snippets.\n"
            f"Original Question: {question}\n"
            "New Query (only output the query, nothing else):",
            lang=lang, style="work", fallback=question,
        ).strip('"')
        k_half = max(1, passage_k // 2)
        k_rest = max(1, passage_k - k_half)
        passages_original = wikipedia_search_passages(question, k=k_half, lang=wiki_lang)
        passages_expanded = wikipedia_search_passages(expanded_query, k=k_rest, lang=wiki_lang)
        passages_all = passages_original + passages_expanded

        bullets = "\n".join([f"- {p.title}: {p.snippet}" for p in passages_all[:12]])
        note = _safe_generate(
            llm_client,
            "Given a question and Wikipedia snippets, write a compact note with only useful info.\n"
            f"Question: {question}\nSnippets:\n{bullets}\nNote:",
            lang=lang, style="work", fallback=bullets,
        )
        ans = _safe_generate(
            llm_client,
            "Answer the question using the note. Give the shortest possible final answer — "
            "ideally one noun phrase — plus one brief sentence.\n"
            f"Question: {question}\nNote: {note}\nAnswer:",
            lang=lang, style="final",
            fallback=("Bilinmiyor" if lang == "tr" else "Unknown"),
        )

        return {
            "mode": mode,
            "question": question,
            "expanded_query": expanded_query,
            "passages_original": [p.__dict__ for p in passages_original],
            "passages_expanded": [p.__dict__ for p in passages_expanded],
            "passage_note": note,
            "answer": ans,
        }

    if mode != "kg_rag":
        raise ValueError("mode must be one of: no_retrieval, vanilla_rag, vanilla_qe, kg_rag")

    # ── KG-Infused RAG ────────────────────────────────────────────────────
    cfg = Neo4jConfig.from_env()
    neo = Neo4jClient(cfg)
    try:
        neo.verify()

        # If the LLM isn't reachable, we still try a pure Neo4j path answer
        # (which is enough for the templated cinema questions).
        try:
            llm_client = _llm_client(llm)
        except Exception:
            llm_client = None

        if llm_client is None:
            path = kg_path_answer(neo, question)
            if path is not None:
                return {
                    "mode": mode,
                    "question": question,
                    "answer": format_answer_for_lang(path.answer, lang),
                    "deterministic_path": {
                        "template": path.template,
                        "answer": path.answer,
                        "answer_id": path.answer_id,
                        "relation_path": path.relation_path,
                        "resolved_mentions": path.resolved_mentions,
                        "entity_path": path.entity_path,
                    },
                    "notice": "LLM unavailable — answered from KG only.",
                }
            # Otherwise bubble up so the UI shows a clear error.
            raise RuntimeError(
                "LLM client could not be initialised and this question does not "
                "match a template that can be answered from the KG alone."
            )

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
        raise SystemExit("Usage: python query_runner.py \"your question\" [kg_rag|vanilla_rag|no_retrieval|vanilla_qe]")
    q = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) >= 3 else "kg_rag"
    out = answer_question(q, mode=mode)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
