import os
import re
from pathlib import Path

from neo4j import GraphDatabase


def _load_env_file(path: Path) -> None:
    """
    Minimal .env loader supporting:
      - KEY=VALUE
      - PowerShell style: $env:KEY = "VALUE"
    Existing environment variables are not overwritten.
    """

    if not path.exists():
        return

    ps_re = re.compile(r'^\s*\$env:(?P<k>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<v>.+?)\s*$')
    kv_re = re.compile(r'^\s*(?P<k>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<v>.+?)\s*$')

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = ps_re.match(line) or kv_re.match(line)
        if not m:
            continue

        k = m.group("k")
        v = m.group("v").strip()

        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            v = v[1:-1]

        os.environ.setdefault(k, v)


_load_env_file(Path(__file__).with_name(".env"))

URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
USER = os.getenv("NEO4J_USER", "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD")


def main() -> None:
    if not PASSWORD:
        raise SystemExit(
            "NEO4J_PASSWORD is not set. In PowerShell run:\n"
            "$env:NEO4J_PASSWORD='your_password_here'\n"
            "Optionally:\n"
            "$env:NEO4J_URI='bolt://localhost:7687'\n"
            "$env:NEO4J_USER='neo4j'"
        )

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        with driver.session() as session:
            rows = session.run("MATCH (n:Entity) RETURN n LIMIT 10").data()
            print(f"rows={len(rows)}")
            for i, row in enumerate(rows, start=1):
                # row["n"] is a neo4j.graph.Node; printing shows labels + properties
                print(f"{i}. {row['n']}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()

