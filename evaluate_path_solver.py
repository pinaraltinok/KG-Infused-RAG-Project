"""Evaluate the deterministic KG path solver on the Türkiye-cinema question set.

This is the Phase 5 evaluator for Module 1 (KG-guided path selection). It
bypasses the LLM entirely and just measures how often the template-based
solver in `kg_path_answer.py` recovers the gold answer from Neo4j.

Usage:
    python evaluate_path_solver.py [N_per_template]

Defaults to 3 questions per template. Pass 50 (or more) to run the whole
dataset.
"""
import json
import sys
import time
from pathlib import Path
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from neo4j_client import Neo4jClient, Neo4jConfig
from kg_path_answer import classify_question, kg_path_answer, _fold


def _norm(s: str) -> str:
    return _fold(s or "").strip()


def _match(pred: str, gold: str) -> bool:
    p, g = _norm(pred), _norm(gold)
    if not p or not g:
        return False
    if p == g or p in g or g in p:
        return True
    pt, gt = set(p.split()), set(g.split())
    if not pt or not gt:
        return False
    return len(pt & gt) / max(len(pt), len(gt)) >= 0.5


QPATH = Path("neo4j_query_table_data_2026-4-16.json")
data = json.loads(QPATH.read_text(encoding="utf-8"))

# Pick a subset covering every template, up to N per template
N_PER_TEMPLATE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
bucket: dict[str, list] = defaultdict(list)
for row in data:
    q = row.get("question")
    if not q:
        continue
    intent = classify_question(q)
    tpl = intent.template if intent else "unclassified"
    if len(bucket[tpl]) < N_PER_TEMPLATE:
        bucket[tpl].append(row)

sample = [r for rows in bucket.values() for r in rows]
print(f"Sample size: {len(sample)} covering {len(bucket)} templates")

neo = Neo4jClient(Neo4jConfig.from_env())
neo.verify()

total = 0
matched = 0
missed = []
per_template: dict[str, list[int]] = defaultdict(lambda: [0, 0])

try:
    for i, row in enumerate(sample):
        q = row.get("question")
        gold = row.get("cevap") or row.get("ortak_tur") or row.get("cevap_film") or ""
        if not q or not gold:
            continue
        total += 1
        t0 = time.perf_counter()
        try:
            res = kg_path_answer(neo, q)
        except Exception as e:
            missed.append({"q": q, "gold": gold, "pred": None, "err": f"{type(e).__name__}: {e}"})
            intent = classify_question(q)
            tpl = intent.template if intent else "unclassified"
            per_template[tpl][1] += 1
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"[{i+1}/{len(sample)}] {elapsed:6.0f}ms  ERR {e}  | {q[:80]}")
            continue

        elapsed = (time.perf_counter() - t0) * 1000
        intent = classify_question(q)
        tpl = intent.template if intent else "unclassified"
        per_template[tpl][1] += 1

        if res is None:
            missed.append({"q": q, "gold": gold, "pred": None, "err": "no_result"})
            print(f"[{i+1}/{len(sample)}] {elapsed:6.0f}ms  MISS (no path)  | {q[:80]}")
            continue
        ok = _match(res.answer, gold)
        if ok:
            matched += 1
            per_template[tpl][0] += 1
            print(f"[{i+1}/{len(sample)}] {elapsed:6.0f}ms  OK  pred={res.answer!r}")
        else:
            missed.append({
                "q": q, "gold": gold, "pred": res.answer,
                "template": res.template,
                "resolved": res.resolved_mentions,
            })
            print(f"[{i+1}/{len(sample)}] {elapsed:6.0f}ms  WRONG pred={res.answer!r}  gold={gold!r}  | {q[:80]}")
finally:
    neo.close()

print(f"\n=== KG path solver accuracy on {total} questions ===")
if total:
    print(f"Correct: {matched}/{total}  ({100.0*matched/total:.1f}%)\n")
print("Per template (correct / total):")
for tpl, (c, t) in sorted(per_template.items(), key=lambda x: -x[1][1]):
    pct = (100.0 * c / t) if t else 0
    print(f"  {tpl}: {c}/{t}  ({pct:.1f}%)")
