from __future__ import annotations

import argparse
import json
import sys

from neo4j_client import Neo4jClient, Neo4jConfig
from spreading_activation import GeminiLLM, OllamaLLM

from kg_infused_rag import KGInfusedRAG


def main() -> None:
    # Make Turkish characters print correctly on Windows terminals where possible.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True)
    ap.add_argument("--domain", default=None, help="football|cinema|company|music|academia")
    ap.add_argument("--llm", choices=["ollama", "gemini"], default="ollama")
    ap.add_argument("--seed-k", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--entities-per-round", type=int, default=10)
    ap.add_argument("--wiki-lang", default="en")
    ap.add_argument("--passage-k", type=int, default=8)
    args = ap.parse_args()

    cfg = Neo4jConfig.from_env()
    neo = Neo4jClient(cfg)
    try:
        neo.verify()
        llm = OllamaLLM() if args.llm == "ollama" else GeminiLLM()
        rag = KGInfusedRAG(
            neo,
            llm=llm,
            domain=args.domain,
            seed_k=args.seed_k,
            max_rounds=args.rounds,
            entities_per_round=args.entities_per_round,
            wiki_lang=args.wiki_lang,
            passage_k=args.passage_k,
        )
        result = rag.answer(args.q)
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    finally:
        neo.close()


if __name__ == "__main__":
    main()

