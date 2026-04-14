from __future__ import annotations

import argparse

from alias_db import build_text_db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-tsv", required=True, help="Path to wikidata5m_text.txt (entityId\\tfull_text)")
    ap.add_argument("--out", default="wikidata_text.db")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = build_text_db(args.text_tsv, out_db_path=args.out, overwrite=bool(args.overwrite))
    print(f"built: {out}")


if __name__ == "__main__":
    main()

