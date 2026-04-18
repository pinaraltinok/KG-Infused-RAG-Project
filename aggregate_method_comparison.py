"""
Merge per-mode evaluation summaries into method_comparison.json for the UI.

Run evaluation once per mode with --write-summary, then:

  python aggregate_method_comparison.py

Defaults read from ./outputs/eval_<mode>.json (see --help for paths).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Keys = evaluation.py --mode values; values = display names in static/index.html table
MODE_ORDER: list[tuple[str, str]] = [
    ("no_retrieval", "No-Retrieval"),
    ("vanilla_rag", "Vanilla RAG"),
    ("vanilla_qe", "Vanilla QE"),
    ("kg_rag", "KG-Infused RAG"),
]


def _load_summary(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def build_method_comparison(
    summaries_by_mode: dict[str, dict[str, Any]],
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Turn dict mode -> summary JSON (from evaluation.py --write-summary) into UI payload."""
    methods: list[dict[str, Any]] = []
    for mode, display_name in MODE_ORDER:
        s = summaries_by_mode.get(mode)
        if not s:
            methods.append(
                {
                    "name": display_name,
                    "acc": None,
                    "f1": None,
                    "em": None,
                    "recall": None,
                }
            )
            continue
        rr = s.get("retrieval_recall")
        methods.append(
            {
                "name": display_name,
                "acc": s.get("acc"),
                "f1": s.get("f1"),
                "em": s.get("em"),
                "recall": None if mode != "kg_rag" else rr,
            }
        )
    out: dict[str, Any] = {"methods": methods}
    if meta:
        out["_meta"] = meta
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build method_comparison.json from eval summaries.")
    ap.add_argument(
        "--out",
        default="method_comparison.json",
        help="Output path (default: method_comparison.json in cwd)",
    )
    ap.add_argument(
        "--summary-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory containing eval_<mode>.json files (default: ./outputs)",
    )
    ap.add_argument("--no-retrieval", type=Path, default=None, help="Override path for no_retrieval summary")
    ap.add_argument("--vanilla-rag", type=Path, default=None, help="Override path for vanilla_rag summary")
    ap.add_argument("--vanilla-qe", type=Path, default=None, help="Override path for vanilla_qe summary")
    ap.add_argument("--kg-rag", type=Path, default=None, help="Override path for kg_rag summary")
    ap.add_argument(
        "--meta-note",
        default="",
        help="Optional _meta.note string (e.g. dataset name and date)",
    )
    args = ap.parse_args()

    paths: dict[str, Path] = {
        "no_retrieval": args.no_retrieval or args.summary_dir / "eval_no_retrieval.json",
        "vanilla_rag": args.vanilla_rag or args.summary_dir / "eval_vanilla_rag.json",
        "vanilla_qe": args.vanilla_qe or args.summary_dir / "eval_vanilla_qe.json",
        "kg_rag": args.kg_rag or args.summary_dir / "eval_kg_rag.json",
    }

    summaries: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for mode, p in paths.items():
        if not p.exists():
            missing.append(f"{mode}: {p}")
            continue
        summaries[mode] = _load_summary(p)

    if missing:
        raise SystemExit(
            "Missing summary file(s):\n  "
            + "\n  ".join(missing)
            + "\n\nRun evaluation.py per mode with --write-summary, or pass explicit paths."
        )

    meta: dict[str, Any] | None = None
    if args.meta_note:
        meta = {"note": args.meta_note}
    elif summaries:
        n = next(iter(summaries.values())).get("n")
        meta = {
            "note": "Aggregated from evaluation.py --write-summary outputs.",
            "n_questions": n,
        }

    payload = build_method_comparison(summaries, meta=meta)
    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"written": str(out_path.resolve()), "modes": list(summaries.keys())}, indent=2))


if __name__ == "__main__":
    main()
