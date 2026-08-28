"""Neo4j persistence with Cypher parameters only - no generated Cypher for ingestion."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from neo4j import GraphDatabase

from .config import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USERNAME


def canonical(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


@lru_cache
def driver():
    if not (NEO4J_URI and NEO4J_PASSWORD):
        raise RuntimeError("NEO4J_URI and NEO4J_PASSWORD must be configured")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def initialize() -> None:
    with driver().session() as session:
        session.run("CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE (e.key, e.type) IS UNIQUE")
        session.run("CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE")


def persist(document: dict[str, Any], extraction: dict[str, Any]) -> dict:
    doc_id = document["id"]
    entities = [{**entity, "key": canonical(entity["name"]), "type": entity["type"].upper().strip()} for entity in extraction["entities"]]
    entity_by_name = {canonical(entity["name"]): entity for entity in entities}
    with driver().session() as session:
        session.run("""MERGE (d:Document {id:$id}) SET d.title=$title, d.source_url=$source_url,
          d.source_type=$source_type, d.domain=$domain, d.author=$author, d.published_at=$published_at,
          d.ingested_at=datetime(), d.text=$text""", **document)
        session.run("""UNWIND $entities AS entity
          MERGE (e:Entity {key:entity.key, type:entity.type})
          ON CREATE SET e.name=entity.name, e.description=entity.description, e.created_at=datetime()
          ON MATCH SET e.description=coalesce(e.description, entity.description)
          WITH e, entity MATCH (d:Document {id:$document_id}) MERGE (d)-[:MENTIONS]->(e)""",
          entities=entities, document_id=doc_id)
        created = 0
        for rel in extraction["relations"]:
            source, target = entity_by_name.get(canonical(rel["source"])), entity_by_name.get(canonical(rel["target"]))
            if not source or not target:
                continue
            # Relationship types cannot be parameterized. The extractor output is constrained then sanitized.
            predicate = re.sub(r"[^A-Z0-9_]", "_", rel["predicate"].upper())[:80] or "RELATED_TO"
            result = session.run(f"""MATCH (s:Entity {{key:$source_key, type:$source_type}}),
              (t:Entity {{key:$target_key, type:$target_type}}), (d:Document {{id:$document_id}})
              MERGE (s)-[r:`{predicate}` {{document_id:$document_id}}]->(t)
              SET r.confidence=$confidence, r.evidence=$evidence, r.created_at=datetime(),
                  r.source_document_id=$document_id
              RETURN count(r) AS count""",
              source_key=source["key"], source_type=source["type"], target_key=target["key"], target_type=target["type"],
              document_id=doc_id, confidence=rel["confidence"], evidence=rel["evidence"])
            created += result.single()["count"]
    return {"document_id": doc_id, "entities_upserted": len(entities), "relations_created": created}


def graph(limit: int = 100, document_id: str | None = None) -> dict:
    with driver().session() as session:
        if document_id:
            nodes = [dict(record) for record in session.run("""MATCH (:Document {id:$document_id})-[:MENTIONS]->(e:Entity)
                RETURN elementId(e) AS id, e.name AS name, e.type AS type, e.description AS description LIMIT $limit""",
                document_id=document_id, limit=limit)]
            edges = [dict(record) for record in session.run("""MATCH (s:Entity)-[r]->(t:Entity)
                WHERE r.document_id=$document_id
                RETURN elementId(r) AS id, elementId(s) AS source_id, elementId(t) AS target_id, type(r) AS predicate,
                       r.confidence AS confidence, r.evidence AS evidence, r.document_id AS document_id LIMIT $limit""",
                document_id=document_id, limit=limit)]
        else:
            nodes = [dict(record) for record in session.run("MATCH (e:Entity) RETURN elementId(e) AS id, e.name AS name, e.type AS type, e.description AS description LIMIT $limit", limit=limit)]
            edges = [dict(record) for record in session.run("MATCH (s:Entity)-[r]->(t:Entity) RETURN elementId(r) AS id, elementId(s) AS source_id, elementId(t) AS target_id, type(r) AS predicate, r.confidence AS confidence, r.evidence AS evidence, r.document_id AS document_id LIMIT $limit", limit=limit)]
    return {"nodes": nodes, "edges": edges}


def list_documents() -> list[dict]:
    with driver().session() as session:
        return [dict(record) for record in session.run("""MATCH (d:Document)
            RETURN d.id AS id, d.title AS title, d.source_type AS source_type, d.ingested_at AS ingested_at
            ORDER BY d.ingested_at DESC""")]


def clear_graph() -> int:
    """Remove every node and relationship from this configured Neo4j database."""
    with driver().session() as session:
        count = session.run("MATCH (node) RETURN count(node) AS count").single()["count"]
        session.run("MATCH (node) DETACH DELETE node").consume()
        return count


def search(query: str) -> list[dict]:
    with driver().session() as session:
        return [dict(row) for row in session.run("MATCH (e:Entity) WHERE toLower(e.name) CONTAINS toLower($query) RETURN elementId(e) AS id, e.name AS name, e.type AS type, e.description AS description LIMIT 50", query=query)]


def neighborhood(entity_id: str) -> dict:
    with driver().session() as session:
        node = session.run("MATCH (e:Entity) WHERE elementId(e)=$id RETURN elementId(e) AS id, e.name AS name, e.type AS type, e.description AS description", id=entity_id).single()
        if not node:
            return {}
        edges = [dict(row) for row in session.run("MATCH (e:Entity)-[r]-(other:Entity) WHERE elementId(e)=$id RETURN elementId(r) AS id, elementId(startNode(r)) AS source_id, elementId(endNode(r)) AS target_id, type(r) AS predicate, r.confidence AS confidence, r.evidence AS evidence", id=entity_id)]
    return {"node": dict(node), "edges": edges}


def read_cypher(query: str) -> list[dict]:
    normalized = query.strip().upper()
    if not (normalized.startswith("MATCH") or normalized.startswith("CALL")) or any(word in normalized for word in ("CREATE", "MERGE", "DELETE", "SET", "DROP", "LOAD CSV")):
        raise ValueError("Only read-only MATCH or CALL queries are allowed")
    with driver().session() as session:
        return [dict(row) for row in session.run(query)]
