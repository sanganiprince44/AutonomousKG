"""REST API for building and querying a provenance-aware knowledge graph."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from . import graph_store
from .config import OLLAMA_HOST, OLLAMA_MODEL
from .evaluation import score
from .preprocessing import chunk_text, clean_text, profile

app = FastAPI(title="Autonomous Knowledge Graph", version="1.0.0")


class DocumentIn(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    text: str = Field(min_length=20, max_length=300_000)
    source_url: str | None = None
    source_type: str = Field(default="text", pattern="^(text|pdf|article|research_paper|news|video_transcript)$")
    domain: str | None = None
    author: str | None = None
    published_at: str | None = None


class Entity(BaseModel):
    name: str = Field(min_length=1, max_length=300)
    type: str = Field(min_length=1, max_length=80)
    description: str | None = Field(default=None, max_length=1_000)


class Relation(BaseModel):
    source: str
    target: str
    predicate: str = Field(min_length=1, max_length=100)
    confidence: float = Field(ge=0, le=1)
    evidence: str = Field(min_length=1, max_length=1_000)


class Extraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


class EvaluationIn(BaseModel):
    predicted: Extraction
    expected: Extraction


class CypherIn(BaseModel):
    query: str = Field(min_length=7, max_length=5_000)


INSTRUCTIONS = """Extract a factual knowledge graph from the supplied document chunk. Return JSON only:
{"entities":[{"name":"...","type":"PERSON|ORG|PLACE|PRODUCT|CONCEPT|EVENT|DATE|OTHER","description":"optional"}],"relations":[{"source":"exact entity name","target":"exact entity name","predicate":"UPPER_SNAKE_CASE","confidence":0.0,"evidence":"short exact supporting quote"}]}
Only include claims explicitly supported by the text. All relationship endpoints must be listed entities. Use specific predicates such as WORKS_FOR, FOUNDED, LOCATED_IN, CREATED, PART_OF, ANNOUNCED."""


@app.on_event("startup")
def startup() -> None:
    try:
        graph_store.initialize()
    except RuntimeError:
        # Allows /health and /docs to remain available before credentials are supplied.
        pass


async def extract_chunk(text: str) -> Extraction:
    """Extract one chunk through a local Ollama model - no paid API is used."""
    schema = Extraction.model_json_schema()
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": (
                "Extract the facts below. Return only JSON that exactly matches this JSON schema. "
                f"Schema: {json.dumps(schema)}\n\nDocument:\n{text}"
            )},
        ],
        "format": schema, "stream": False,
        "options": {"temperature": 0, "num_predict": 4_000},
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
    except httpx.RequestError as error:
        raise HTTPException(503, f"Ollama is not running at {OLLAMA_HOST}. Install Ollama, then run: ollama pull {OLLAMA_MODEL}") from error
    if response.status_code >= 400:
        raise HTTPException(502, f"Ollama extraction failed: {response.text[:300]}")
    try:
        output = response.json()["message"]["content"]
        return Extraction.model_validate(json.loads(output))
    except (KeyError, IndexError, ValueError) as error:
        raise HTTPException(502, f"Local model returned an invalid extraction: {error}") from error


async def extract_document(text: str) -> Extraction:
    entities: dict[tuple[str, str], Entity] = {}
    relations: list[Relation] = []
    for chunk in chunk_text(text):
        extraction = await extract_chunk(chunk)
        for entity in extraction.entities:
            entities.setdefault((entity.name.strip().casefold(), entity.type.strip().upper()), entity)
        relations.extend(extraction.relations)
    entity_names = {entity.name.strip().casefold() for entity in entities.values()}
    valid_relations = [rel for rel in relations if rel.source.strip().casefold() in entity_names and rel.target.strip().casefold() in entity_names]
    return Extraction(entities=list(entities.values()), relations=valid_relations)


def graph_error(error: Exception) -> HTTPException:
    return HTTPException(503, f"Graph database unavailable: {error}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "graph_configured": bool(__import__("os").getenv("NEO4J_URI")), "llm_provider": "ollama", "ollama_model": OLLAMA_MODEL}


@app.post("/documents", status_code=201)
async def ingest_document(document: DocumentIn) -> dict:
    cleaned = clean_text(document.text)
    extraction = await extract_document(cleaned)
    record = document.model_dump() | {"id": str(uuid4()), "text": cleaned, "ingested_at": datetime.now(timezone.utc).isoformat()}
    try:
        result = graph_store.persist(record, extraction.model_dump())
    except Exception as error:
        raise graph_error(error) from error
    return {**result, "profile": profile(cleaned), "extraction": extraction.model_dump()}


@app.get("/graph")
def graph(limit: int = Query(default=100, ge=1, le=1000), document_id: str | None = None) -> dict:
    try:
        return graph_store.graph(limit, document_id)
    except Exception as error:
        raise graph_error(error) from error


@app.get("/entities/search")
def search_entities(q: str = Query(min_length=1, max_length=100)) -> list[dict]:
    try:
        return graph_store.search(q)
    except Exception as error:
        raise graph_error(error) from error


@app.get("/entities/{entity_id}/neighborhood")
def neighborhood(entity_id: str) -> dict:
    try:
        result = graph_store.neighborhood(entity_id)
    except Exception as error:
        raise graph_error(error) from error
    if not result:
        raise HTTPException(404, "Entity not found")
    return result


@app.post("/evaluate")
def evaluate(payload: EvaluationIn) -> dict:
    return {"entities": score([item.model_dump() for item in payload.predicted.entities], [item.model_dump() for item in payload.expected.entities], ("name", "type")),
            "relations": score([item.model_dump() for item in payload.predicted.relations], [item.model_dump() for item in payload.expected.relations], ("source", "predicate", "target"))}


@app.post("/cypher")
def cypher(payload: CypherIn) -> list[dict]:
    try:
        return graph_store.read_cypher(payload.query)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    except Exception as error:
        raise graph_error(error) from error
