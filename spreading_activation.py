from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import unicodedata
import urllib.request
from typing import Any, Iterable, Protocol

from neo4j_client import Neo4jClient
from alias_db import alias_db_path_from_env, query_aliases, query_text, text_db_path_from_env
from llm_language import detect_language, wrap_prompt


@dataclass(frozen=True)
class SeedEntity:
    id: str
    name: str | None
    score: float


@dataclass(frozen=True)
class Triple:
    source_id: str
    source_name: str | None
    relation_type: str
    relation_id: str | None
    target_id: str
    target_name: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": {"id": self.source_id, "name": self.source_name},
            "relation": {"type": self.relation_type, "id": self.relation_id},
            "target": {"id": self.target_id, "name": self.target_name},
        }

    def to_llm_string(self) -> str:
        s = self.source_name or self.source_id
        t = self.target_name or self.target_id
        return f"<{s} | {self.relation_type} | {t}>"


class TripleSelector(Protocol):
    def select(self, query: str, triples: list[Triple]) -> list[Triple]: ...


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    # Common Turkish casing normalization (İ/ı edge cases).
    s = s.replace("İ", "i").replace("I", "i").replace("ı", "i")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_query_variants(query: str, *, domain: str | None = None) -> list[str]:
    q0 = query.strip()
    qn = _normalize_text(q0)

    variants = [q0, qn]
    # Türkiye/Turkey normalization boosts recall.
    if "turkiye" in qn or "türkiye" in q0.lower() or "turkey" in qn:
        variants.extend(["Türkiye", "Turkiye", "Turkey", "turkiye", "turkey"])

    domain_tokens: dict[str, list[str]] = {
        "football": ["football", "soccer", "futbol", "club", "kulup", "coach", "manager", "player"],
        "cinema": ["film", "movie", "cinema", "director", "actor", "award", "festival"],
        "company": ["company", "bank", "airlines", "holding", "headquarters", "founder", "industry"],
        "music": ["music", "singer", "album", "band", "record label", "genre"],
        "academia": ["university", "educated", "professor", "research", "field of work"],
    }
    if domain:
        toks = domain_tokens.get(domain.lower().strip(), [])
        # Append a few tokens to help fulltext scoring without over-constraining.
        if toks:
            base = q0 if len(q0) < 120 else q0[:120]
            variants.append(base + " " + " ".join(toks[:4]))

    # Deduplicate while preserving order
    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        v = v.strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


class NoFilterSelector:
    """Return all triples (bounded elsewhere)."""

    def select(self, query: str, triples: list[Triple]) -> list[Triple]:
        _ = query
        return triples


class ScoreTargetSelector:
    """
    Non-LLM baseline: keep triples whose target is in the top-N scored targets.
    """

    def __init__(self, *, max_entities: int, decay: float):
        self.max_entities = max_entities
        self.decay = decay

    def select_with_source_activation(
        self,
        triples: list[Triple],
        source_activation: dict[str, float],
        visited: set[str],
    ) -> tuple[list[Triple], list[str], dict[str, float]]:
        target_scores: dict[str, float] = {}
        for t in triples:
            src_score = source_activation.get(t.source_id, 0.0)
            target_scores[t.target_id] = target_scores.get(t.target_id, 0.0) + (src_score * self.decay)

        ranked_targets = sorted(
            ((tid, sc) for tid, sc in target_scores.items() if tid not in visited),
            key=lambda x: x[1],
            reverse=True,
        )
        next_entities = [tid for tid, _ in ranked_targets[: self.max_entities]]
        selected_set = set(next_entities)
        selected_triples = [t for t in triples if t.target_id in selected_set]
        return selected_triples, next_entities, target_scores

    def select(self, query: str, triples: list[Triple]) -> list[Triple]:
        # This selector needs activation context; call select_with_source_activation instead.
        _ = query
        return triples


class OpenAICompatibleLLM:
    """
    Minimal OpenAI-compatible Chat Completions client.
    Uses only stdlib (no `openai` dependency).

    Env:
      - OPENAI_API_KEY (required)
      - OPENAI_BASE_URL (optional, default https://api.openai.com)
      - OPENAI_MODEL (optional, default gpt-4o-mini)
    """

    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # Many local OpenAI-compatible servers (and Ollama) don't require a key.
        # If you're hitting a hosted endpoint, set OPENAI_API_KEY.
        if not self.api_key and not (
            self.base_url.startswith("http://localhost")
            or self.base_url.startswith("http://127.0.0.1")
            or self.base_url.startswith("http://0.0.0.0")
        ):
            raise RuntimeError("OPENAI_API_KEY is not set (needed for LLM triple filtering).")

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only what the user asked for. Be terse."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
            },
            method="POST",
        )
        if self.api_key:
            req.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return parsed["choices"][0]["message"]["content"]


class OllamaLLM:
    """
    Ollama chat endpoint client.

    Env:
      - OLLAMA_HOST (optional, default http://localhost:11434)
      - OLLAMA_MODEL (optional, default qwen2.5:7b-instruct)
    """

    def __init__(self):
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        # People often set Ollama to an OpenAI-compatible base like http://localhost:11434/v1
        # Normalize so we can call /api/chat reliably.
        if host.endswith("/v1"):
            host = host[: -len("/v1")]
        if host.endswith("/api"):
            host = host[: -len("/api")]
        self.host = host
        # Use the common Ollama tag naming. Override with OLLAMA_MODEL if needed.
        self.model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.timeout_s = float(os.getenv("OLLAMA_TIMEOUT_S", "30"))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "80"))

    def generate(self, prompt: str) -> str:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": self.num_predict},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8")
        parsed = json.loads(body)
        return parsed["message"]["content"]


class GeminiLLM:
    """
    Google Gemini API (fast hosted alternative to Ollama).

    Env:
      - GEMINI_API_KEY (required)
      - GEMINI_MODEL (optional, default gemini-1.5-flash)
    """

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.timeout_s = float(os.getenv("GEMINI_TIMEOUT_S", "30"))
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

    def generate(self, prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        parsed = json.loads(body)
        # typical shape: candidates[0].content.parts[0].text
        cands = parsed.get("candidates") or []
        if not cands:
            return ""
        parts = ((cands[0].get("content") or {}).get("parts")) or []
        if not parts:
            return ""
        return str(parts[0].get("text") or "")


class LLMTripleSelector:
    """
    Tutorial Step 6.5: ask an LLM to pick relevant triples by index.
    """

    def __init__(self, llm: OpenAICompatibleLLM):
        self.llm = llm

    def select(self, query: str, triples: list[Triple]) -> list[Triple]:
        # Keep the prompt bounded; large triple lists can make local LLMs "hang" for a long time.
        # This cap is intentionally small—selection happens every round.
        max_triples = int(os.getenv("LLM_TRIPLE_CAP", "24"))
        if len(triples) > max_triples:
            triples = triples[:max_triples]
        triple_lines = [f"{i}: {t.to_llm_string()}" for i, t in enumerate(triples)]
        lang = detect_language(query)
        prompt = (
            "Given a question and a set of entity triples, select only the triples that are relevant "
            "to answering the question.\n"
            f"Question: {query}\n"
            "Triples:\n"
            + "\n".join(triple_lines)
            + "\nReturn ONLY the indices of relevant triples, separated by commas. "
            "If no triples are relevant, return \"NONE\"."
        )
        raw = self.llm.generate(wrap_prompt(prompt, lang=lang)).strip()
        if "NONE" in raw.upper():
            return []

        # Extract integers robustly (handles "0, 2, 5" or "Indices: 0,2")
        idxs = [int(x) for x in re.findall(r"\d+", raw)]
        out: list[Triple] = []
        for i in idxs:
            if 0 <= i < len(triples):
                out.append(triples[i])
        return out


def find_seed_entities_keyword(
    neo: Neo4jClient,
    query: str,
    *,
    k: int = 5,
    fulltext_index: str = "entity_search",
    domain: str | None = None,
) -> list[SeedEntity]:
    variants = _build_query_variants(query, domain=domain)

    # We keep strict precedence by assigning score bands:
    # exact name >> alias DB >> text DB >> Neo4j fulltext >> fallback CONTAINS
    best: dict[str, SeedEntity] = {}

    # Prefer multi-word proper nouns (e.g., "Nuri Bilge Ceylan") over single tokens (e.g., "Nuri").
    raw_tokens = [t for t in re.split(r"\s+", query.strip()) if t]
    phrases: list[str] = []
    cur: list[str] = []
    for tok in raw_tokens:
        t = tok.strip(".,;:!?()[]{}\"'“”‘’")
        if not t:
            continue
        if t[:1].isupper() and any(ch.isalpha() for ch in t):
            cur.append(t)
        else:
            if len(cur) >= 2:
                phrases.append(" ".join(cur))
            cur = []
    if len(cur) >= 2:
        phrases.append(" ".join(cur))

    mention_candidates = re.findall(r"\b[^\W\d_][\w'’\-]{3,}\b", query, flags=re.UNICODE)
    single_mentions = [m for m in mention_candidates if m[:1].isupper() and len(m) >= 4]

    # Deduplicate while keeping phrase priority.
    mentions: list[tuple[str, float]] = []
    seen_m: set[str] = set()
    for p in phrases:
        if p not in seen_m:
            mentions.append((p, 1000.0))
            seen_m.add(p)
    for m in single_mentions:
        if m not in seen_m:
            mentions.append((m, 150.0))
            seen_m.add(m)

    # 1) Exact name match in Neo4j (highest precision)
    for m, _boost in mentions[:6]:
        rows = neo.run(
            """
            MATCH (e:Entity)
            WHERE toLower(coalesce(e.name,'')) = toLower($m)
            RETURN e.entityId AS id, e.name AS name
            LIMIT 10
            """,
            {"m": m},
        )
        for r in rows:
            if r.get("id") is None:
                continue
            eid = str(r["id"])
            ent = SeedEntity(id=eid, name=(None if r.get("name") is None else str(r.get("name"))), score=10_000.0)
            prev = best.get(eid)
            if prev is None or ent.score > prev.score:
                best[eid] = ent

    # Alias DB boost (Option B): if you built `wikidata_aliases.db`, use it for fast alias-based linking.
    alias_db = alias_db_path_from_env()
    if alias_db is not None and alias_db.exists():
        # Query with a few high-signal tokens (proper nouns or longer tokens).
        tokens = [m for (m, _boost) in mentions[:5]]
        if not tokens:
            tokens = [t for t in re.split(r"\s+", query) if len(t) >= 5][:5]
        for tok in tokens:
            hits = query_aliases(alias_db, tok, k=20)
            for h in hits:
                ent = SeedEntity(id=h.entity_id, name=None, score=8_000.0 + (h.score * 500.0))
                prev = best.get(ent.id)
                if prev is None or ent.score > prev.score:
                    best[ent.id] = ent

    # Text DB boost (Option B): if you built `wikidata_text.db` from wikidata5m_text.txt,
    # use it to link entities by description text.
    text_db = text_db_path_from_env()
    if text_db is not None and text_db.exists():
        # Use the whole query (FTS5 will pick terms), plus a normalized variant.
        for qv in {query.strip(), _normalize_text(query)}:
            if not qv:
                continue
            hits = query_text(text_db, qv, k=30)
            for h in hits:
                ent = SeedEntity(id=h.entity_id, name=None, score=6_000.0 + (h.score * 500.0))
                prev = best.get(ent.id)
                if prev is None or ent.score > prev.score:
                    best[ent.id] = ent

    # 4) Neo4j fulltext (name+description), aggregated across variants
    per_variant_k = max(k, 5)
    for qv in variants:
        rows = neo.run(
            """
            CALL db.index.fulltext.queryNodes($index, $q)
            YIELD node, score
            RETURN node.entityId AS id, node.name AS name, score
            ORDER BY score DESC
            LIMIT $k
            """,
            {"index": fulltext_index, "q": qv, "k": per_variant_k},
        )
        for r in rows:
            if r.get("id") is None:
                continue
            eid = str(r["id"])
            ent = SeedEntity(
                id=eid,
                name=(r.get("name") if r.get("name") is None else str(r.get("name"))),
                score=2_000.0 + float(r["score"]),
            )
            prev = best.get(eid)
            if prev is None or ent.score > prev.score:
                best[eid] = ent

    # 5) Fallback CONTAINS (lowest precision)
    if not best:
        qn = _normalize_text(query)
        rows = neo.run(
            """
            MATCH (e:Entity)
            WHERE toLower(coalesce(e.name,'')) CONTAINS toLower($q)
            RETURN e.entityId AS id, e.name AS name
            LIMIT $k
            """,
            {"q": qn, "k": int(k)},
        )
        for r in rows:
            if r.get("id") is None:
                continue
            eid = str(r["id"])
            best[eid] = SeedEntity(id=eid, name=(None if r.get("name") is None else str(r.get("name"))), score=1.0)

    ranked = sorted(best.values(), key=lambda s: s.score, reverse=True)

    # Türkiye-domain re-ranking/filtering (assignment focus).
    # If TURKEY_ENTITY_ID is present in your graph (typically Q43), we prefer seeds connected by :COUNTRY.
    turkey_id = os.getenv("TURKEY_ENTITY_ID", "Q43")
    candidate_ids = [s.id for s in ranked[: max(30, k * 10)]]
    if candidate_ids:
        rows = neo.run(
            """
            UNWIND $ids AS id
            MATCH (e:Entity {entityId: id})
            OPTIONAL MATCH (e)-[:COUNTRY]->(t:Entity {entityId: $turkey})
            RETURN id AS id, count(t) AS c
            """,
            {"ids": candidate_ids, "turkey": turkey_id},
        )
        turkey_connected: set[str] = {str(r["id"]) for r in rows if int(r.get("c", 0)) > 0}
        if turkey_connected:
            ranked = sorted(
                ranked,
                key=lambda s: (s.id in turkey_connected, s.score),
                reverse=True,
            )

    return ranked[:k]


def get_one_hop_neighbors(
    neo: Neo4jClient,
    entity_ids: Iterable[str],
    visited_ids: Iterable[str],
    *,
    per_source_limit: int = 50,
) -> list[Triple]:
    """
    Matches tutorial Step 6.3, but with a per-source LIMIT to keep each round bounded.
    """
    rows = neo.run(
        """
        UNWIND $entity_ids AS eid
        MATCH (e:Entity {entityId: eid})-[r]->(neighbor:Entity)
        WHERE NOT neighbor.entityId IN $visited_ids
        RETURN
          e.entityId AS source_id,
          e.name AS source_name,
          type(r) AS relation_type,
          r.relationId AS relation_id,
          neighbor.entityId AS target_id,
          neighbor.name AS target_name
        LIMIT $lim
        """,
        {"entity_ids": list(entity_ids), "visited_ids": list(visited_ids), "lim": int(per_source_limit) * max(1, len(list(entity_ids)))},
    )

    triples: list[Triple] = []
    for r in rows:
        if r.get("source_id") is None or r.get("target_id") is None or r.get("relation_type") is None:
            continue
        triples.append(
            Triple(
                source_id=str(r["source_id"]),
                source_name=(None if r.get("source_name") is None else str(r.get("source_name"))),
                relation_type=str(r["relation_type"]),
                relation_id=(None if r.get("relation_id") is None else str(r.get("relation_id"))),
                target_id=str(r["target_id"]),
                target_name=(None if r.get("target_name") is None else str(r.get("target_name"))),
            )
        )
    return triples


class SpreadingActivation:
    """
    A minimal, tutorial-faithful spreading activation loop:
      - seed retrieval via fulltext
      - 1-hop neighbor expansion
      - visited set
      - 2–3 rounds (configurable)

    The tutorial uses an LLM to select relevant triples; by default we keep the
    highest-activated targets (simple score propagation) so you can run it now.
    """

    def __init__(
        self,
        neo: Neo4jClient,
        *,
        selector: TripleSelector | None = None,
        max_rounds: int = 3,
        max_entities_per_round: int = 10,
        max_triples_per_round: int = 200,
        max_triples_per_entity: int = 20,
        decay: float = 0.85,
    ):
        self.neo = neo
        self.selector = selector
        self.max_rounds = max_rounds
        self.max_entities_per_round = max_entities_per_round
        self.max_triples_per_round = max_triples_per_round
        self.max_triples_per_entity = max_triples_per_entity
        self.decay = decay
        if self.selector is None:
            # Default to non-LLM baseline (still bounded + repeatable).
            self.selector = ScoreTargetSelector(max_entities=max_entities_per_round, decay=decay)

    def _limit_triples(self, triples: list[Triple]) -> list[Triple]:
        """
        Tutorial 6.4 mentions limiting triples; we cap per-source to avoid a single hub exploding the round.
        """
        if self.max_triples_per_entity <= 0:
            return triples
        per_source: dict[str, int] = {}
        out: list[Triple] = []
        for t in triples:
            c = per_source.get(t.source_id, 0)
            if c >= self.max_triples_per_entity:
                continue
            per_source[t.source_id] = c + 1
            out.append(t)
        return out

    def run(self, query: str, seed_entity_ids: list[str]) -> dict[str, Any]:
        visited: set[str] = set()
        subgraph: list[dict[str, Any]] = []
        trace_rounds: list[dict[str, Any]] = []

        activation: dict[str, float] = {eid: 1.0 for eid in seed_entity_ids}
        current_entities: list[str] = list(dict.fromkeys(seed_entity_ids))

        # baseline selector can optionally use activation context
        score_selector = self.selector if isinstance(self.selector, ScoreTargetSelector) else None
        fallback_score_selector = ScoreTargetSelector(max_entities=self.max_entities_per_round, decay=self.decay)

        for round_idx in range(self.max_rounds):
            if not current_entities:
                break
            round_current = list(current_entities)

            triples = get_one_hop_neighbors(self.neo, current_entities, visited)
            if not triples:
                break
            triples = self._limit_triples(triples)

            selected_triples: list[Triple]
            next_entities: list[str]
            used_fallback = False

            if score_selector is not None:
                selected_triples, next_entities, target_scores = score_selector.select_with_source_activation(
                    triples, activation, visited
                )
            else:
                # LLM (or other) selector: choose relevant triples, then next entities are the targets.
                try:
                    selected_triples = self.selector.select(query, triples)
                except Exception:
                    # If the LLM call times out / fails, don't kill the whole run.
                    selected_triples, next_entities, target_scores = fallback_score_selector.select_with_source_activation(
                        triples, activation, visited
                    )
                    used_fallback = True
                    if len(selected_triples) > self.max_triples_per_round:
                        selected_triples = selected_triples[: self.max_triples_per_round]
                    subgraph.extend([t.as_dict() for t in selected_triples])
                    visited.update(current_entities)
                    ranked_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
                    for tid, sc in ranked_targets[: self.max_entities_per_round]:
                        activation[tid] = max(activation.get(tid, 0.0), float(sc))
                    current_entities = [e for e in next_entities if e not in visited]
                    trace_rounds.append(
                        {
                            "round": round_idx + 1,
                            "current_entities": round_current,
                            "triples_considered": len(triples),
                            "selected_triples": [t.as_dict() for t in selected_triples],
                            "next_entities": list(next_entities),
                            "selector_fallback_used": True,
                        }
                    )
                    continue
                target_scores = {}
                next_entities = list(dict.fromkeys([t.target_id for t in selected_triples if t.target_id not in visited]))[
                    : self.max_entities_per_round
                ]

            if not selected_triples:
                # LLM may answer "NONE" too aggressively; fall back to score-based selection for this round.
                selected_triples, next_entities, target_scores = fallback_score_selector.select_with_source_activation(
                    triples, activation, visited
                )
                used_fallback = True
                if not selected_triples:
                    break

            if len(selected_triples) > self.max_triples_per_round:
                selected_triples = selected_triples[: self.max_triples_per_round]

            subgraph.extend([t.as_dict() for t in selected_triples])

            visited.update(current_entities)
            # update activation scores for next targets if we have them
            if target_scores:
                ranked_targets = sorted(target_scores.items(), key=lambda x: x[1], reverse=True)
                for tid, sc in ranked_targets[: self.max_entities_per_round]:
                    activation[tid] = max(activation.get(tid, 0.0), float(sc))
            else:
                for tid in next_entities:
                    activation[tid] = max(activation.get(tid, 0.0), 0.0)

            current_entities = [e for e in next_entities if e not in visited]

            trace_rounds.append(
                {
                    "round": round_idx + 1,
                    "current_entities": round_current,
                    "triples_considered": len(triples),
                    "selected_triples": [t.as_dict() for t in selected_triples],
                    "next_entities": list(next_entities),
                    "selector_fallback_used": bool(used_fallback),
                }
            )

        # return top activated entities for debugging / downstream use
        top_activated = sorted(activation.items(), key=lambda x: x[1], reverse=True)[:50]
        return {
            "query": query,
            "seeds": seed_entity_ids,
            "visited": sorted(visited),
            "subgraph": subgraph,
            "trace_rounds": trace_rounds,
            "top_activated": [{"id": eid, "activation": score} for eid, score in top_activated],
        }

