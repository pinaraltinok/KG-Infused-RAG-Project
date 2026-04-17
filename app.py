from __future__ import annotations

import time
import traceback
import hashlib
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from neo4j.exceptions import AuthError, ServiceUnavailable
from neo4j_client import Neo4jClient, Neo4jConfig
from query_runner import answer_question
from spreading_activation import SpreadingActivation, find_seed_entities_keyword


app = FastAPI(title="SocialRAG", version="0.3")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────
# Global exception handlers
# ──────────────────────────────────────────

@app.exception_handler(ServiceUnavailable)
async def _neo4j_unavailable_handler(_: Request, exc: ServiceUnavailable):
    return JSONResponse(
        status_code=503,
        content={
            "error": "neo4j_unavailable",
            "message": (
                "Neo4j is unavailable. Start Neo4j and ensure Bolt is listening on "
                "localhost:7687, or set NEO4J_URI/NEO4J_USER/NEO4J_PASSWORD in .env."
            ),
            "detail": str(exc),
        },
    )


@app.exception_handler(AuthError)
async def _neo4j_auth_handler(_: Request, exc: AuthError):
    return JSONResponse(
        status_code=401,
        content={
            "error": "neo4j_auth_failed",
            "message": "Neo4j authentication failed. Check NEO4J_USER/NEO4J_PASSWORD in .env.",
            "detail": str(exc),
        },
    )


@app.exception_handler(Exception)
async def _generic_handler(_: Request, exc: Exception):
    tb = traceback.format_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "message": str(exc),
            "traceback": tb[-2000:],  # truncate for safety
        },
    )


# ──────────────────────────────────────────
# Request / response models
# ──────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    mode: str = Field("kg_rag")   # kg_rag|vanilla_rag|vanilla_qe|no_retrieval
    llm: str = Field("ollama")    # ollama|gemini
    domain: str | None = None
    rounds: int = 3
    seed_k: int = 5
    entities_per_round: int = 10
    wiki_lang: str = "tr"
    passage_k: int = 8


class CypherTemplateRequest(BaseModel):
    template_id: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=200)
    q: str | None = None


class EvalComparisonRequest(BaseModel):
    questions_path: str | None = None


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def _with_neo4j(fn):
    cfg = Neo4jConfig.from_env()
    neo = Neo4jClient(cfg)
    try:
        neo.verify()
        return fn(neo)
    finally:
        neo.close()


def _query_templates() -> dict[str, dict]:
    return {
        "turkiye_root_entity": {
            "title": "Türkiye Root Entity",
            "cypher": """
            MATCH (e:Entity)
            WHERE toLower(coalesce(e.name,'')) CONTAINS 'turkey'
               OR toLower(coalesce(e.name,'')) CONTAINS 'türkiye'
            RETURN e.entityId AS id, e.name AS name
            LIMIT $limit
            """,
            "params": ["limit"],
        },
        "turkish_cities": {
            "title": "Turkish Cities Detection",
            "cypher": """
            MATCH (city:Entity)-[:COUNTRY]->(:Entity {entityId: 'Q43'})
            MATCH (city)-[:INSTANCE_OF]->(type:Entity)
            WHERE toLower(coalesce(type.name,'')) CONTAINS 'city'
            RETURN city.entityId AS id, city.name AS city, type.name AS type
            ORDER BY city
            LIMIT $limit
            """,
            "params": ["limit"],
        },
        "football_players_club": {
            "title": "Football Players → Club",
            "cypher": """
            MATCH (player:Entity)-[r]->(club:Entity)
            WHERE (type(r) IN ['MEMBER_OF_SPORTS_TEAM', 'PLAYS_FOR'] OR r.relationId = 'P54')
              AND toLower(coalesce(club.name,'')) CONTAINS toLower($q)
            RETURN player.entityId AS player_id, player.name AS player, club.name AS club, type(r) AS rel
            LIMIT $limit
            """,
            "params": ["q", "limit"],
            "default_q": "Galatasaray",
        },
        "coach_birth_place": {
            "title": "Coach → Birth Place",
            "cypher": """
            MATCH (club:Entity)-[rc]->(coach:Entity)
            WHERE (type(rc) IN ['COACH', 'HEAD_COACH', 'MANAGER'] OR rc.relationId = 'P286')
              AND toLower(coalesce(club.name,'')) CONTAINS toLower($q)
            OPTIONAL MATCH (coach)-[rb]->(birth:Entity)
            WHERE (type(rb) IN ['PLACE_OF_BIRTH', 'BIRTH_PLACE'] OR rb.relationId = 'P19')
            RETURN club.name AS club, coach.name AS coach, birth.name AS birth_place
            LIMIT $limit
            """,
            "params": ["q", "limit"],
            "default_q": "Galatasaray",
        },
        "film_director_award": {
            "title": "Film → Director → Award",
            "cypher": """
            MATCH (film:Entity)-[:DIRECTOR]->(director:Entity)
            WHERE toLower(coalesce(film.name,'')) CONTAINS toLower($q)
            OPTIONAL MATCH (director)-[:AWARD_RECEIVED]->(award:Entity)
            RETURN film.name AS film, director.name AS director, collect(DISTINCT award.name)[0..8] AS awards
            LIMIT $limit
            """,
            "params": ["q", "limit"],
            "default_q": "Kış Uykusu",
        },
        "club_stadium_city": {
            "title": "Club → Stadium → City",
            "cypher": """
            MATCH (club:Entity)-[rv]->(stadium:Entity)
            WHERE (type(rv) IN ['HOME_VENUE'] OR rv.relationId = 'P115')
              AND toLower(coalesce(club.name,'')) CONTAINS toLower($q)
            OPTIONAL MATCH (stadium)-[rl]->(city:Entity)
            WHERE (type(rl) IN ['LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY'] OR rl.relationId = 'P131')
            RETURN club.name AS club, stadium.name AS stadium, city.name AS city
            LIMIT $limit
            """,
            "params": ["q", "limit"],
            "default_q": "Fenerbahçe",
        },
        "relation_frequency_turkiye": {
            "title": "Relation Frequency (Türkiye Context)",
            "cypher": """
            MATCH (e:Entity)-[:COUNTRY]->(:Entity {entityId: 'Q43'})
            MATCH (e)-[r]->()
            RETURN type(r) AS relation, count(*) AS frequency
            ORDER BY frequency DESC
            LIMIT $limit
            """,
            "params": ["limit"],
        },
        "film_narrative_location": {
            "title": "Film → Narrative Location → Country",
            "cypher": """
            MATCH (film:Entity)-[:NARRATIVE_LOCATION]->(loc:Entity)
            WHERE toLower(coalesce(film.name,'')) CONTAINS toLower($q)
            OPTIONAL MATCH (loc)-[:COUNTRY]->(ctr:Entity)
            RETURN film.name AS film, loc.name AS location, ctr.name AS country
            LIMIT $limit
            """,
            "params": ["q", "limit"],
            "default_q": "İçerde",
        },
    }


def _infer_entity_type(neo: Neo4jClient, entity_id: str) -> str:
    rows = neo.run(
        """
        MATCH (e:Entity {entityId:$id})-[:INSTANCE_OF]->(t:Entity)
        RETURN toLower(coalesce(t.name,'')) AS t
        LIMIT 8
        """,
        {"id": entity_id},
    )
    types = [str(r.get("t") or "") for r in rows]
    if any("football club" in t or "club" in t for t in types):
        return "CLUB"
    if any("city" in t for t in types):
        return "CITY"
    if any("human" in t or "person" in t or "football player" in t for t in types):
        return "PERSON"
    if any("company" in t or "organization" in t or "railway" in t for t in types):
        return "ORG"
    if any("film" in t for t in types):
        return "FILM"
    return "OTHER"


# ──────────────────────────────────────────
# Dashboard / DB endpoints
# ──────────────────────────────────────────

@app.get("/api/db/stats")
def api_db_stats():
    def _run(neo: Neo4jClient):
        node_count = int(neo.run("MATCH (n:Entity) RETURN count(n) AS c")[0]["c"])
        rel_count = int(neo.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"])
        turkiye_connected = int(
            neo.run(
                "MATCH (e:Entity)-[:COUNTRY]->(:Entity {entityId: 'Q43'}) RETURN count(DISTINCT e) AS c"
            )[0]["c"]
        )
        return {
            "entity_count": node_count,
            "triple_count": rel_count,
            "turkiye_connected_entities": turkiye_connected,
        }
    return _with_neo4j(_run)


@app.get("/api/ui/knowledge_overview")
def api_ui_knowledge_overview():
    def _run(neo: Neo4jClient):
        turkey_entities = int(
            neo.run("MATCH (e:Entity)-[:COUNTRY]->(:Entity {entityId:'Q43'}) RETURN count(DISTINCT e) AS c")[0]["c"]
        )
        total_relations = int(neo.run("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"])

        # bounded 2-hop count for responsive UI
        two_hop = int(
            neo.run(
                """
                MATCH (a:Entity)-[:COUNTRY]->(:Entity {entityId:'Q43'})
                WITH a LIMIT 5000
                MATCH (a)-[]->(:Entity)-[]->(:Entity)
                RETURN count(*) AS c
                """
            )[0]["c"]
        )
        domains = api_domain_distribution()
        rel_rows = api_relation_frequency(limit=10)
        return {
            "turkey_entities": turkey_entities,
            "total_relations": total_relations,
            "two_hop_paths": two_hop,
            "domains_count": len(domains),
            "domain_distribution": domains,
            "top_relations": rel_rows,
        }

    return _with_neo4j(_run)


@app.get("/api/db/seed_entities")
def api_seed_entities(
    query: str = "Galatasaray coach birth place", k: int = 8, domain: str | None = None
):
    def _run(neo: Neo4jClient):
        seeds = find_seed_entities_keyword(neo, query, k=k, domain=domain)
        out = []
        for s in seeds:
            deg_row = neo.run(
                "MATCH (e:Entity {entityId:$id})-[r]->() RETURN count(r) AS c",
                {"id": s.id},
            )
            out.append(
                {
                    "id": s.id,
                    "name": s.name,
                    "type": _infer_entity_type(neo, s.id),
                    "score": float(s.score),
                    "relations": int((deg_row[0]["c"] if deg_row else 0)),
                }
            )
        return out
    return _with_neo4j(_run)


@app.get("/api/db/relation_frequency")
def api_relation_frequency(limit: int = 20):
    def _run(neo: Neo4jClient):
        rows = neo.run(
            """
            MATCH (e:Entity)-[:COUNTRY]->(:Entity {entityId:'Q43'})
            MATCH (e)-[r]->()
            RETURN type(r) AS relation, count(*) AS frequency
            ORDER BY frequency DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [{"relation": str(r["relation"]), "frequency": int(r["frequency"])} for r in rows]
    return _with_neo4j(_run)


@app.get("/api/db/sample_triples")
def api_sample_triples(limit: int = 10):
    def _run(neo: Neo4jClient):
        rows = neo.run(
            """
            MATCH (s:Entity)-[r]->(t:Entity)
            WHERE r.relationId IS NOT NULL
            RETURN s.entityId AS source_id, r.relationId AS relation_id, t.entityId AS target_id
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [
            {
                "source_id": str(r["source_id"]),
                "relation_id": str(r["relation_id"]),
                "target_id": str(r["target_id"]),
            }
            for r in rows
        ]
    return _with_neo4j(_run)


@app.get("/api/db/query_templates")
def api_query_templates():
    tpls = _query_templates()
    return [
        {
            "id": k,
            "title": v["title"],
            "params": v.get("params", []),
            "default_q": v.get("default_q"),
        }
        for k, v in tpls.items()
    ]


@app.get("/api/ui/cypher_metrics")
def api_ui_cypher_metrics():
    templates = _query_templates()

    def _run(neo: Neo4jClient):
        total_ms = 0.0
        total_rows = 0
        for _, tpl in templates.items():
            params: dict = {"limit": 20}
            if "q" in tpl.get("params", []):
                params["q"] = tpl.get("default_q", "")
            t0 = time.perf_counter()
            rows = neo.run(tpl["cypher"], params)
            total_ms += (time.perf_counter() - t0) * 1000.0
            total_rows += len(rows)
        avg_ms = total_ms / max(1, len(templates))
        return {
            "query_templates": len(templates),
            "avg_exec_time_ms": round(avg_ms, 1),
            "max_hops": 3,
            "result_rows": total_rows,
        }

    return _with_neo4j(_run)


@app.get("/api/ui/embedding_projection")
def api_ui_embedding_projection(limit: int = 120):
    def _run(neo: Neo4jClient):
        rows = neo.run(
            """
            MATCH (e:Entity)-[:COUNTRY]->(:Entity {entityId:'Q43'})
            RETURN e.entityId AS id, e.name AS name
            LIMIT $limit
            """,
            {"limit": limit},
        )
        points = []
        for r in rows:
            eid = str(r["id"])
            typ = _infer_entity_type(neo, eid)
            # deterministic pseudo projection from entity id hash (stable per project data)
            h = int(hashlib.md5(eid.encode("utf-8")).hexdigest()[:8], 16)
            x = ((h % 10000) / 10000.0) * 2 - 1
            y = (((h // 10000) % 10000) / 10000.0) * 2 - 1
            points.append({"id": eid, "name": r.get("name"), "type": typ, "x": x, "y": y})
        return {"points": points}

    return _with_neo4j(_run)


@app.get("/api/ui/method_comparison")
def api_ui_method_comparison():
    """
    If a precomputed comparison JSON exists in project root, use it.
    Otherwise return shape with null values so frontend still works.
    """
    root = Path(".")
    candidates = [
        root / "method_comparison.json",
        root / "results" / "method_comparison.json",
        root / "outputs" / "method_comparison.json",
    ]
    for p in candidates:
        if p.exists():
            try:
                data = p.read_text(encoding="utf-8")
                return {"available": True, "source": str(p), "data": __import__("json").loads(data)}
            except Exception:
                pass
    return {
        "available": False,
        "source": None,
        "data": {
            "methods": [
                {"name": "No-Retrieval", "acc": None, "f1": None, "em": None, "recall": None},
                {"name": "Vanilla RAG", "acc": None, "f1": None, "em": None, "recall": None},
                {"name": "Vanilla QE", "acc": None, "f1": None, "em": None, "recall": None},
                {"name": "KG-Infused RAG", "acc": None, "f1": None, "em": None, "recall": None},
            ]
        },
        "message": "No precomputed method_comparison.json found yet.",
    }


@app.post("/api/db/run_template")
def api_run_template(req: CypherTemplateRequest):
    templates = _query_templates()
    tpl = templates.get(req.template_id)
    if not tpl:
        return JSONResponse(
            status_code=404,
            content={"error": "not_found", "message": f"Unknown template_id: {req.template_id}"},
        )

    def _run(neo: Neo4jClient):
        params: dict = {"limit": req.limit}
        if "q" in tpl.get("params", []):
            params["q"] = req.q if req.q else tpl.get("default_q", "")
        rows = neo.run(tpl["cypher"], params)
        return {
            "template_id": req.template_id,
            "title": tpl["title"],
            "cypher": tpl["cypher"].strip(),
            "params": params,
            "rows": rows,
            "row_count": len(rows),
        }

    return _with_neo4j(_run)


@app.get("/api/db/spreading_activation_preview")
def api_spreading_preview(
    question: str = "Galatasaray takımının teknik direktörünün doğum yeri neresidir?",
    seed_k: int = 5,
    rounds: int = 3,
    entities_per_round: int = 10,
):
    def _run(neo: Neo4jClient):
        seeds = find_seed_entities_keyword(neo, question, k=seed_k, domain="football")
        seed_ids = [s.id for s in seeds]
        sa = SpreadingActivation(
            neo,
            selector=None,
            max_rounds=rounds,
            max_entities_per_round=entities_per_round,
        )
        t0 = time.perf_counter()
        act = sa.run(question, seed_ids)
        ms = int((time.perf_counter() - t0) * 1000)
        return {
            "question": question,
            "seed_entities": [s.__dict__ for s in seeds],
            "round_count": len(act.get("trace_rounds", [])),
            "subgraph_triple_count": len(act.get("subgraph", [])),
            "trace_rounds": act.get("trace_rounds", []),
            "top_activated": act.get("top_activated", []),
            "elapsed_ms": ms,
        }
    return _with_neo4j(_run)


@app.get("/api/db/domain_distribution")
def api_domain_distribution():
    def _run(neo: Neo4jClient):
        rows = neo.run(
            """
            MATCH (e:Entity)-[:COUNTRY]->(:Entity {entityId:'Q43'})
            OPTIONAL MATCH (e)-[:INSTANCE_OF]->(t:Entity)
            WITH e, collect(toLower(coalesce(t.name,''))) AS types
            WITH
              CASE
                WHEN any(x IN types WHERE x CONTAINS 'football') THEN 'football'
                WHEN any(x IN types WHERE x CONTAINS 'film') THEN 'cinema'
                WHEN any(x IN types WHERE x CONTAINS 'company' OR x CONTAINS 'airline' OR x CONTAINS 'bank') THEN 'company'
                WHEN any(x IN types WHERE x CONTAINS 'song' OR x CONTAINS 'album' OR x CONTAINS 'music') THEN 'music'
                WHEN any(x IN types WHERE x CONTAINS 'university' OR x CONTAINS 'academic') THEN 'academia'
                ELSE 'other'
              END AS domain
            RETURN domain, count(*) AS count
            ORDER BY count DESC
            """
        )
        return [{"domain": str(r["domain"]), "count": int(r["count"])} for r in rows]
    return _with_neo4j(_run)


# ──────────────────────────────────────────
# QA endpoints  — hardened with try/except
# ──────────────────────────────────────────

@app.post("/api/ask")
def api_ask(req: AskRequest):
    try:
        result = answer_question(
            req.question,
            mode=req.mode,
            llm=req.llm,
            domain=req.domain,
            rounds=req.rounds,
            seed_k=req.seed_k,
            entities_per_round=req.entities_per_round,
            wiki_lang=req.wiki_lang,
            passage_k=req.passage_k,
        )
        return result
    except ServiceUnavailable as exc:
        return JSONResponse(
            status_code=503,
            content={
                "error": "neo4j_unavailable",
                "message": "Neo4j is not reachable. Is it running?",
                "detail": str(exc),
            },
        )
    except AuthError as exc:
        return JSONResponse(
            status_code=401,
            content={
                "error": "neo4j_auth_failed",
                "message": "Neo4j authentication failed.",
                "detail": str(exc),
            },
        )
    except RuntimeError as exc:
        # e.g. GEMINI_API_KEY not set, NEO4J_PASSWORD missing
        return JSONResponse(
            status_code=400,
            content={
                "error": "configuration_error",
                "message": str(exc),
            },
        )
    except Exception as exc:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": type(exc).__name__,
                "message": str(exc),
                "traceback": tb[-3000:],
            },
        )


@app.post("/api/answer", response_class=PlainTextResponse)
def api_answer(req: AskRequest):
    try:
        out = answer_question(
            req.question,
            mode=req.mode,
            llm=req.llm,
            domain=req.domain,
            rounds=req.rounds,
            seed_k=req.seed_k,
            entities_per_round=req.entities_per_round,
            wiki_lang=req.wiki_lang,
            passage_k=req.passage_k,
        )
        ans = out.get("answer") if isinstance(out, dict) else str(out)
        if not ans:
            ans = ""
        return str(ans).strip().replace("\r\n", "\n").replace("\r", "\n").strip()
    except Exception as exc:
        return PlainTextResponse(
            status_code=500,
            content=f"ERROR: {type(exc).__name__}: {exc}",
        )


# ──────────────────────────────────────────
# Static + SPA
# ──────────────────────────────────────────

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()
