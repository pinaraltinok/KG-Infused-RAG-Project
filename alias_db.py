from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class AliasHit:
    entity_id: str
    alias: str
    score: float


def build_alias_db(
    alias_tsv_path: str | Path,
    *,
    out_db_path: str | Path = "wikidata_aliases.db",
    overwrite: bool = False,
) -> Path:
    """
    Build a tiny SQLite FTS index for Wikidata5M alias file.

    Expected TSV format (common in Wikidata5M): entityId<TAB>alias1<TAB>alias2...
    """
    alias_tsv_path = Path(alias_tsv_path)
    out_db_path = Path(out_db_path)

    if out_db_path.exists():
        if overwrite:
            out_db_path.unlink()
        else:
            return out_db_path

    con = sqlite3.connect(str(out_db_path))
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")

        con.execute(
            """
            CREATE VIRTUAL TABLE alias_fts USING fts5(
              alias,
              entity_id UNINDEXED
            );
            """
        )

        with alias_tsv_path.open("r", encoding="utf-8", errors="replace") as f:
            batch: list[tuple[str, str]] = []
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                entity_id = parts[0].strip()
                if not entity_id:
                    continue
                for a in parts[1:]:
                    a = a.strip()
                    if not a:
                        continue
                    batch.append((a, entity_id))
                if len(batch) >= 5000:
                    con.executemany("INSERT INTO alias_fts(alias, entity_id) VALUES (?, ?)", batch)
                    con.commit()
                    batch = []
            if batch:
                con.executemany("INSERT INTO alias_fts(alias, entity_id) VALUES (?, ?)", batch)
                con.commit()

        return out_db_path
    finally:
        con.close()


def build_text_db(
    text_tsv_path: str | Path,
    *,
    out_db_path: str | Path = "wikidata_text.db",
    overwrite: bool = False,
) -> Path:
    """
    Build an SQLite FTS index for `wikidata5m_text.txt` style files:
      entityId<TAB>full_text
    """
    text_tsv_path = Path(text_tsv_path)
    out_db_path = Path(out_db_path)

    if out_db_path.exists():
        if overwrite:
            out_db_path.unlink()
        else:
            return out_db_path

    con = sqlite3.connect(str(out_db_path))
    try:
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA temp_store=MEMORY;")

        con.execute(
            """
            CREATE VIRTUAL TABLE text_fts USING fts5(
              text,
              entity_id UNINDEXED
            );
            """
        )

        con.execute(
            """
            CREATE TABLE meta(
              k TEXT PRIMARY KEY,
              v TEXT NOT NULL
            );
            """
        )
        con.execute("INSERT INTO meta(k,v) VALUES ('lines', '0')")
        con.commit()

        with text_tsv_path.open("r", encoding="utf-8", errors="replace") as f:
            batch: list[tuple[str, str]] = []
            total = 0
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    continue
                entity_id = parts[0].strip()
                txt = parts[1].strip()
                if not entity_id or not txt:
                    continue
                batch.append((txt, entity_id))
                total += 1
                if len(batch) >= 2000:
                    con.executemany("INSERT INTO text_fts(text, entity_id) VALUES (?, ?)", batch)
                    con.execute("UPDATE meta SET v=? WHERE k='lines'", (str(total),))
                    con.commit()
                    batch = []
            if batch:
                con.executemany("INSERT INTO text_fts(text, entity_id) VALUES (?, ?)", batch)
                con.execute("UPDATE meta SET v=? WHERE k='lines'", (str(total),))
                con.commit()

        return out_db_path
    finally:
        con.close()


def query_aliases(
    db_path: str | Path,
    query: str,
    *,
    k: int = 20,
) -> list[AliasHit]:
    """
    Query alias FTS. Returns entity_ids + matched alias + bm25 score.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    con = sqlite3.connect(str(db_path))
    try:
        toks = re.findall(r"[^\W\d_]{2,}", query, flags=re.UNICODE)
        if not toks:
            return []
        # FTS5 query: OR tokens, prefix match for recall.
        fts_q = " OR ".join([f'"{t}"*' for t in toks[:8]])
        # bm25() lower is better; we invert to a positive score.
        rows = con.execute(
            """
            SELECT entity_id, alias, bm25(alias_fts) AS rank
            FROM alias_fts
            WHERE alias_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_q, int(k)),
        ).fetchall()
        out: list[AliasHit] = []
        for entity_id, alias, rank in rows:
            try:
                r = float(rank)
            except Exception:
                r = 0.0
            out.append(AliasHit(entity_id=str(entity_id), alias=str(alias), score=1.0 / (1.0 + max(r, 0.0))))
        return out
    finally:
        con.close()


def query_text(
    db_path: str | Path,
    query: str,
    *,
    k: int = 20,
) -> list[AliasHit]:
    """
    Query text FTS. Returns entity_ids + a snippet + bm25-derived score.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        return []

    con = sqlite3.connect(str(db_path))
    try:
        toks = re.findall(r"[^\W\d_]{2,}", query, flags=re.UNICODE)
        if not toks:
            return []
        fts_q = " OR ".join([f'"{t}"*' for t in toks[:12]])
        rows = con.execute(
            """
            SELECT entity_id, snippet(text_fts, 0, '', '', ' … ', 12) AS snip, bm25(text_fts) AS rank
            FROM text_fts
            WHERE text_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (fts_q, int(k)),
        ).fetchall()
        out: list[AliasHit] = []
        for entity_id, snip, rank in rows:
            try:
                r = float(rank)
            except Exception:
                r = 0.0
            out.append(AliasHit(entity_id=str(entity_id), alias=str(snip), score=1.0 / (1.0 + max(r, 0.0))))
        return out
    finally:
        con.close()


def alias_db_path_from_env() -> Path | None:
    p = os.getenv("WIKIDATA_ALIAS_DB", "").strip()
    return Path(p) if p else None


def text_db_path_from_env() -> Path | None:
    p = os.getenv("WIKIDATA_TEXT_DB", "").strip()
    return Path(p) if p else None

