from __future__ import annotations

import argparse

from alias_db import build_alias_db


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias-tsv", required=True, help="Path to wikidata5m alias TSV (entityId\\talias1\\talias2...)")
    ap.add_argument("--out", default="wikidata_aliases.db")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = build_alias_db(args.alias_tsv, out_db_path=args.out, overwrite=bool(args.overwrite))
    print(f"built: {out}")


if __name__ == "__main__":
    main()

