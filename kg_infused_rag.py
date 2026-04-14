from __future__ import annotations

from dataclasses import dataclass

from neo4j_client import Neo4jClient
from retrieval_wikipedia import WikiPassage, wikipedia_search_passages
from spreading_activation import GeminiLLM, LLMTripleSelector, OllamaLLM, SpreadingActivation, find_seed_entities_keyword
from llm_language import detect_language, wrap_prompt


@dataclass(frozen=True)
class KGInfusedRAGResult:
    answer: str
    kg_summary: str
    expanded_query: str
    seed_entities: list[dict]
    activation: dict
    trace: dict
    passages_original: list[dict]
    passages_expanded: list[dict]
    passage_note: str
    enhanced_note: str


class KGInfusedRAG:
    """
    Assignment Phase 4 / tutorial Step 7:
      - Module 1: KG-guided spreading activation (already implemented)
      - Module 2: KG-based query expansion + dual retrieval
      - Module 3: KG-augmented answer generation
    """

    def __init__(
        self,
        neo: Neo4jClient,
        *,
        llm: OllamaLLM | GeminiLLM,
        fulltext_index: str = "entity_search",
        domain: str | None = None,
        seed_k: int = 3,
        max_rounds: int = 3,
        entities_per_round: int = 10,
        wiki_lang: str = "en",
        passage_k: int = 8,
    ):
        self.neo = neo
        self.llm = llm
        self.fulltext_index = fulltext_index
        self.domain = domain
        self.seed_k = seed_k
        self.max_rounds = max_rounds
        self.entities_per_round = entities_per_round
        self.wiki_lang = wiki_lang
        self.passage_k = passage_k

    def _summarize_subgraph(self, query: str, triples: list[dict]) -> str:
        if not triples:
            return "No relevant information found in the knowledge graph."
        lang = detect_language(query)
        facts = []
        for t in triples[:80]:
            s = t["source"]["name"] or t["source"]["id"]
            r = str(t["relation"]["type"]).lower().replace("_", " ")
            o = t["target"]["name"] or t["target"]["id"]
            facts.append(f"- {s} {r} {o}")
        prompt = (
            "Given a question and related facts from a knowledge graph, write a concise summary "
            "capturing the key information. Keep it short (4-8 sentences).\n"
            f"Question: {query}\n"
            "Facts:\n"
            + "\n".join(facts)
            + "\nSummary:"
        )
        return self.llm.generate(wrap_prompt(prompt, lang=lang)).strip()

    def _expand_query(self, original_query: str, kg_summary: str) -> str:
        if not kg_summary or kg_summary.strip().lower().startswith("no relevant information found"):
            # Avoid hallucinating a totally different query when KG summary is empty.
            return original_query
        lang = detect_language(original_query)
        prompt = (
            "Generate a new short query that is distinct from but related to the original. "
            "Use the provided KG information to retrieve additional relevant passages.\n"
            f"Original Question: {original_query}\n"
            f"Related Information: {kg_summary}\n"
            "New Query (only output the query, nothing else):"
        )
        return self.llm.generate(wrap_prompt(prompt, lang=lang)).strip().strip('"')

    def _passage_note(self, query: str, passages: list[WikiPassage]) -> str:
        if not passages:
            return "No passages retrieved."
        lang = detect_language(query)
        bullets = []
        for p in passages[:12]:
            bullets.append(f"- {p.title}: {p.snippet}")
        prompt = (
            "Given a question and a set of retrieved Wikipedia snippets, write a compact note "
            "containing only information helpful for answering the question.\n"
            f"Question: {query}\n"
            "Snippets:\n"
            + "\n".join(bullets)
            + "\nNote:"
        )
        return self.llm.generate(wrap_prompt(prompt, lang=lang)).strip()

    def _augment_with_kg(self, query: str, passage_note: str, kg_summary: str) -> str:
        lang = detect_language(query)
        prompt = (
            "You are an expert at improving a note for QA. Given a question, a note from passages, "
            "and factual information from a knowledge graph, rewrite the note by integrating useful KG facts. "
            "Do not add unrelated info.\n"
            f"Question: {query}\n"
            f"Passage note: {passage_note}\n"
            f"KG facts: {kg_summary}\n"
            "Enhanced note:"
        )
        return self.llm.generate(wrap_prompt(prompt, lang=lang)).strip()

    def _answer(self, query: str, enhanced_note: str) -> str:
        lang = detect_language(query)
        prompt = (
            "Answer the question using the provided note. If the note is insufficient, say what is missing.\n"
            f"Question: {query}\n"
            f"Note: {enhanced_note}\n"
            "Answer:"
        )
        return self.llm.generate(wrap_prompt(prompt, lang=lang)).strip()

    def answer(self, query: str) -> KGInfusedRAGResult:
        # Module 1: seeds + spreading activation (LLM triple selection)
        seeds = find_seed_entities_keyword(
            self.neo,
            query,
            k=self.seed_k,
            fulltext_index=self.fulltext_index,
            domain=self.domain,
        )
        seed_ids = [s.id for s in seeds]

        selector = LLMTripleSelector(self.llm)
        sa = SpreadingActivation(
            self.neo,
            selector=selector,
            max_rounds=self.max_rounds,
            max_entities_per_round=self.entities_per_round,
        )
        activation = sa.run(query, seed_ids)

        kg_summary = self._summarize_subgraph(query, activation.get("subgraph", []))

        # Module 2: KG-based query expansion + dual retrieval
        expanded_query = self._expand_query(query, kg_summary)
        k_half = max(1, self.passage_k // 2)
        passages_original = wikipedia_search_passages(query, k=k_half, lang=self.wiki_lang)
        passages_expanded = wikipedia_search_passages(expanded_query, k=self.passage_k - k_half, lang=self.wiki_lang)
        passages_all = passages_original + passages_expanded

        # Module 3: note -> KG augmentation -> answer
        passage_note = self._passage_note(query, passages_all)
        enhanced_note = self._augment_with_kg(query, passage_note, kg_summary)
        answer = self._answer(query, enhanced_note)

        trace = {
            "question": query,
            "seeds": seed_ids,
            "seed_entities": [s.__dict__ for s in seeds],
            "rounds": activation.get("trace_rounds", []),
            "kg_summary": kg_summary,
            "expanded_query": expanded_query,
        }

        return KGInfusedRAGResult(
            answer=answer,
            kg_summary=kg_summary,
            expanded_query=expanded_query,
            seed_entities=[s.__dict__ for s in seeds],
            activation=activation,
            trace=trace,
            passages_original=[p.__dict__ for p in passages_original],
            passages_expanded=[p.__dict__ for p in passages_expanded],
            passage_note=passage_note,
            enhanced_note=enhanced_note,
        )

