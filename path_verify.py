from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neo4j_client import Neo4jClient, Neo4jConfig


def _parse_path(path: str) -> tuple[list[str], list[str]]:
    # expected: A -> rel1 -> B -> rel2 -> C
    p = path.replace("→", "->")
    parts = [x.strip() for x in p.split("->") if x.strip()]
    entities = [parts[i] for i in range(0, len(parts), 2)]
    rels = [parts[i] for i in range(1, len(parts), 2)]
    return entities, rels


def _match_entity_id(neo: Neo4jClient, name_or_id: str) -> str | None:
    if name_or_id.upper().startswith("Q") and name_or_id[1:].isdigit():
        return name_or_id
    rows = neo.run(
        """
        CALL db.index.fulltext.queryNodes('entity_search', $q)
        YIELD node, score
        RETURN node.entityId AS id
        ORDER BY score DESC
        LIMIT 1
        """,
        {"q": name_or_id},
    )
    return str(rows[0]["id"]) if rows else None


def _relation_matches(rel_type: str, rel_id: str | None, expected: str) -> bool:
    e = expected.strip().upper().replace(" ", "_")
    return rel_type == e or (rel_id or "").upper() == e


def verify_question(neo: Neo4jClient, q: dict[str, Any]) -> dict[str, Any]:
    path = str(q.get("reasoning_path") or "")
    if not path:
        return {"question_id": q.get("question_id"), "ok": False, "reason": "missing_reasoning_path"}
    entities, rels = _parse_path(path)
    if len(entities) < 2 or len(rels) != len(entities) - 1:
        return {"question_id": q.get("question_id"), "ok": False, "reason": "invalid_path_format"}

    entity_ids: list[str] = []
    for e in entities:
        eid = _match_entity_id(neo, e)
        if not eid:
            return {"question_id": q.get("question_id"), "ok": False, "reason": f"entity_not_found:{e}"}
        entity_ids.append(eid)

    for i, rel in enumerate(rels):
        rows = neo.run(
            """
            MATCH (a:Entity {entityId:$a})-[r]->(b:Entity {entityId:$b})
            RETURN type(r) AS rel_type, r.relationId AS rel_id
            LIMIT 20
            """,
            {"a": entity_ids[i], "b": entity_ids[i + 1]},
        )
        if not rows:
            return {"question_id": q.get("question_id"), "ok": False, "reason": f"edge_missing:{i+1}", "rel": rel}
        if not any(_relation_matches(str(r["rel_type"]), (None if r.get("rel_id") is None else str(r["rel_id"])), rel) for r in rows):
            return {"question_id": q.get("question_id"), "ok": False, "reason": f"relation_mismatch:{i+1}", "rel": rel}

    return {"question_id": q.get("question_id"), "ok": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="JSON list or {questions:[...]} with reasoning_path")
    args = ap.parse_args()

    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    qs = data["questions"] if isinstance(data, dict) and "questions" in data else data
    if not isinstance(qs, list):
        raise SystemExit("Invalid questions format.")

    neo = Neo4jClient(Neo4jConfig.from_env())
    try:
        neo.verify()
        out = [verify_question(neo, q) for q in qs if isinstance(q, dict)]
    finally:
        neo.close()

    ok = sum(1 for r in out if r.get("ok"))
    print(json.dumps({"n": len(out), "verified": ok, "failed": len(out) - ok, "details": out}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

