from __future__ import annotations

import argparse
import json

from neo4j_client import Neo4jClient, Neo4jConfig
from spreading_activation import (
    LLMTripleSelector,
    GeminiLLM,
    OllamaLLM,
    OpenAICompatibleLLM,
    SpreadingActivation,
    find_seed_entities_keyword,
 )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="User question / query string")
    ap.add_argument("--seed-k", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--entities-per-round", type=int, default=10)
    ap.add_argument("--selector", choices=["score", "llm", "ollama", "gemini"], default="score")
    args = ap.parse_args()

    cfg = Neo4jConfig.from_env()
    neo = Neo4jClient(cfg)
    try:
        neo.verify()
        seeds = find_seed_entities_keyword(neo, args.q, k=args.seed_k)
        seed_ids = [s.id for s in seeds]
        if not seed_ids:
            raise SystemExit("No seeds found from fulltext index. Check the `entity_search` index exists and has data.")

        selector = None
        if args.selector == "llm":
            selector = LLMTripleSelector(OpenAICompatibleLLM())
        elif args.selector == "ollama":
            selector = LLMTripleSelector(OllamaLLM())
            # Warm up the model so the first "real" selection doesn't look like a freeze.
            try:
                selector.llm.generate("Reply with only: OK")
            except Exception:
                pass
        elif args.selector == "gemini":
            selector = LLMTripleSelector(GeminiLLM())

        sa = SpreadingActivation(
            neo,
            selector=selector,
            max_rounds=args.rounds,
            max_entities_per_round=args.entities_per_round,
        )
        result = sa.run(args.q, seed_ids)
        result["seed_entities"] = [s.__dict__ for s in seeds]

        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        neo.close()


if __name__ == "__main__":
    main()

