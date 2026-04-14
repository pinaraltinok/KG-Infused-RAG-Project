from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable

from neo4j import GraphDatabase

from kg_env import load_env_file


@dataclass(frozen=True)
class Neo4jConfig:
    uri: str
    user: str
    password: str

    @staticmethod
    def from_env() -> "Neo4jConfig":
        load_env_file(".env")
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USER", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        if not password:
            raise RuntimeError("NEO4J_PASSWORD is missing. Add it to `.env` as NEO4J_PASSWORD=...")
        return Neo4jConfig(uri=uri, user=user, password=password)


class Neo4jClient:
    def __init__(self, config: Neo4jConfig):
        self._driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))

    def close(self) -> None:
        self._driver.close()

    def verify(self) -> None:
        self._driver.verify_connectivity()

    def run(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            res = session.run(cypher, parameters or {})
            return [r.data() for r in res]

    def stream(self, cypher: str, parameters: dict[str, Any] | None = None) -> Iterable[dict[str, Any]]:
        with self._driver.session() as session:
            res = session.run(cypher, parameters or {})
            for r in res:
                yield r.data()

