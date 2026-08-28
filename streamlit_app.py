"""Simple visual interface for building and exploring the project graph."""
from __future__ import annotations

import asyncio
import json
import math
from uuid import uuid4

import fitz
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import graph_store
from app.evaluation import score
from app.main import DocumentIn, extract_document
from app.preprocessing import clean_text, profile

st.set_page_config(page_title="Autonomous Knowledge Graph", layout="wide")
st.title("Autonomous Knowledge Graph Construction")
st.caption("Free local Ollama-powered entity and relationship extraction with Neo4j provenance.")
st.session_state.setdefault("evaluation_predicted", '{"entities": [], "relations": []}')
st.session_state.setdefault("evaluation_ground_truth", '{"entities": [], "relations": []}')
st.session_state.setdefault("current_document_id", None)


def extract_pdf(data: bytes) -> str:
    document = fitz.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text() for page in document)


def draw_graph(data: dict) -> None:
    nodes, edges = data["nodes"], data["edges"]
    if not nodes:
        st.info("No graph data yet. Ingest a document first.")
        return
    coords = {node["id"]: (math.cos(2 * math.pi * index / len(nodes)), math.sin(2 * math.pi * index / len(nodes))) for index, node in enumerate(nodes)}
    figure = go.Figure()
    for edge in edges:
        if edge["source_id"] in coords and edge["target_id"] in coords:
            x0, y0 = coords[edge["source_id"]]; x1, y1 = coords[edge["target_id"]]
            figure.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line={"color":"#9ca3af"}, hoverinfo="text", text=edge["predicate"], showlegend=False))
    figure.add_trace(go.Scatter(x=[coords[node["id"]][0] for node in nodes], y=[coords[node["id"]][1] for node in nodes], mode="markers+text", text=[node["name"] for node in nodes], textposition="top center", marker={"size":15, "color":"#2563eb"}, hovertext=[node["type"] for node in nodes], showlegend=False))
    figure.update_layout(height=550, margin={"l": 0, "r": 0, "t": 20, "b": 0}, xaxis={"visible": False}, yaxis={"visible": False})
    st.plotly_chart(figure)


ingest, explore, evaluate = st.tabs(["Ingest document", "Explore graph", "Evaluate extraction"])
with ingest:
    upload = st.file_uploader("Upload a PDF or TXT file", type=["pdf", "txt"])
    text = st.text_area("Or paste text", height=220, placeholder="Paste an article, research-paper abstract, report, or transcript...")
    title = st.text_input("Document title")
    source_url = st.text_input("Source URL (optional)")
    domain = st.text_input("Domain/category (optional)")
    source_type = "text"
    if upload:
        source_type = "pdf" if upload.type == "application/pdf" else "text"
        text = extract_pdf(upload.getvalue()) if source_type == "pdf" else upload.getvalue().decode("utf-8", errors="replace")
        title = title or upload.name
    if text:
        stats = profile(text)
        st.write({key: value for key, value in stats.items() if key != "top_terms"})
        st.bar_chart(pd.DataFrame(stats["top_terms"], columns=["term", "count"]).set_index("term"))
    if st.button("Construct knowledge graph", type="primary", disabled=not (title and text)):
        try:
            cleaned = clean_text(text)
            with st.spinner("Cleaning, chunking, extracting triples locally, and storing in Neo4j..."):
                extraction = asyncio.run(extract_document(cleaned))
                record = DocumentIn(title=title, text=cleaned, source_url=source_url or None, source_type=source_type, domain=domain or None).model_dump() | {"id": str(uuid4()), "text": cleaned}
                result = graph_store.persist(record, extraction.model_dump())
            extraction_json = json.dumps(extraction.model_dump(), indent=2)
            st.session_state["evaluation_predicted"] = extraction_json
            st.session_state["current_document_id"] = result["document_id"]
            st.success(f"Stored {result['entities_upserted']} entities and {result['relations_created']} relationships.")
            st.caption("The evaluator has been pre-filled with this valid JSON.")
            st.code(extraction_json, language="json")
            st.json(extraction.model_dump())
        except Exception as error:
            st.error(str(error))
with explore:
    try:
        documents = graph_store.list_documents()
    except Exception as error:
        documents = []
        st.error(f"Could not load documents: {error}")
    scope = st.segmented_control(
        "Graph scope", ["Current document", "All documents"],
        default="Current document", selection_mode="single",
    )
    current_document_id = st.session_state["current_document_id"]
    if scope == "Current document" and not current_document_id and documents:
        current_document_id = documents[0]["id"]
    if scope == "Current document" and not current_document_id:
        st.info("Ingest a document first, then its graph can be viewed here.")
    if st.button("Load selected graph", type="primary"):
        try:
            document_id = current_document_id if scope == "Current document" else None
            draw_graph(graph_store.graph(100, document_id))
        except Exception as error:
            st.error(str(error))
    if documents:
        st.caption("Stored documents")
        st.dataframe(pd.DataFrame(documents))
    if scope == "Current document" and current_document_id:
        query_default = f"""MATCH (s:Entity)-[r]->(t:Entity)
WHERE r.document_id = '{current_document_id}'
RETURN DISTINCT s.name, type(r), t.name, r.evidence
LIMIT 25"""
        st.caption("This query is filtered to the current document.")
    else:
        query_default = "MATCH (s:Entity)-[r]->(t:Entity) RETURN DISTINCT s.name, type(r), t.name, r.evidence LIMIT 25"
        st.caption("This query includes all stored documents.")
    query = st.text_area("Read-only Cypher query", value=query_default, height=130)
    if st.button("Run Cypher"):
        try:
            st.dataframe(pd.DataFrame(graph_store.read_cypher(query)))
        except Exception as error:
            st.error(str(error))
    with st.expander("Reset demo graph"):
        st.warning("This permanently removes every document, entity, and relationship in the configured Neo4j database.")
        confirmed = st.checkbox("I understand that this cannot be undone", key="confirm_graph_reset")
        if st.button("Clear entire graph", disabled=not confirmed):
            try:
                removed = graph_store.clear_graph()
                st.session_state["current_document_id"] = None
                st.success(f"Removed {removed} graph nodes. You can now start with a clean graph.")
            except Exception as error:
                st.error(f"Could not clear graph: {error}")
with evaluate:
    st.write("The latest extraction is automatically available below as valid JSON. Paste a manually labelled ground truth to calculate precision, recall, and F1.")
    predicted = st.text_area("Predicted JSON", height=260, key="evaluation_predicted")
    expected = st.text_area("Ground-truth JSON", height=260, key="evaluation_ground_truth")
    if st.button("Calculate metrics"):
        try:
            predicted_data, expected_data = json.loads(predicted), json.loads(expected)
            st.json({"entities": score(predicted_data["entities"], expected_data["entities"], ("name", "type")), "relations": score(predicted_data["relations"], expected_data["relations"], ("source", "predicate", "target"))})
        except Exception as error:
            st.error(f"Invalid evaluation data: {error}")
