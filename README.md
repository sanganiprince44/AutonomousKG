# Autonomous Knowledge Graph Construction

An application that converts unstructured documents into a provenance-aware Neo4j knowledge graph.

## What it does

1. Accepts a document through an HTTP API.
2. Uses a free, local Ollama model to extract canonical entities and typed relationships as structured JSON.
3. Cleans and chunks the text, then validates the extraction and generates subject-predicate-object triples.
4. Stores de-duplicated entities, relationships, source evidence, and document metadata in Neo4j.
5. Provides a Streamlit interface for upload, EDA, graph visualization, and safe read-only Cypher queries.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Fill the Neo4j credentials in .env. No OpenAI API key is needed.
# Install Ollama from https://ollama.com/download, then in a terminal run:
ollama pull llama3.2:3b
uvicorn app.main:app --reload

# In a second terminal
streamlit run streamlit_app.py
```

Open `http://127.0.0.1:8000/docs` for the interactive API.

## Example

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H 'content-type: application/json' \
  -d '{"title":"Example","text":"Ada Lovelace worked with Charles Babbage on the Analytical Engine."}'

curl 'http://127.0.0.1:8000/graph?limit=100'
```

## API

- `POST /documents` — preprocess text, extract triples, and construct graph facts.
- `GET /graph` — return nodes and edges.
- `GET /entities/search?q=ada` — find canonical entities.
- `GET /entities/{id}/neighborhood` — retrieve a one-hop subgraph.
- `POST /evaluate` — calculate entity/relation precision, recall, and F1 against labelled triples.
- `POST /cypher` — run a read-only Cypher query.
- `GET /health` — health check.

## Architecture notes

The graph preserves the original text span and confidence for every fact. Entity identity is normalized by `(name, type)` and relationships are de-duplicated by `(source, predicate, target, document)`. The application deliberately runs only read-only Cypher from the UI. It uses a locally downloaded Ollama model, so document content stays on the Mac and no paid LLM API is called.
