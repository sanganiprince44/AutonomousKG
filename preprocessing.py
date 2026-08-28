"""Text extraction, cleaning, chunking, and lightweight EDA."""
from __future__ import annotations

import re
from collections import Counter


STOP_WORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "is", "are", "for", "on", "with", "as", "by", "from", "that", "this", "it"}


def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, max_chars: int = 5_000) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(text))
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 > max_chars and current:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    return chunks + ([current] if current else [])


def profile(text: str) -> dict:
    cleaned = clean_text(text)
    words = re.findall(r"[A-Za-z][A-Za-z'-]+", cleaned.lower())
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", cleaned) if s]
    counts = Counter(word for word in words if word not in STOP_WORDS)
    return {
        "characters": len(cleaned), "words": len(words), "sentences": len(sentences),
        "average_sentence_length": round(len(words) / len(sentences), 1) if sentences else 0,
        "top_terms": counts.most_common(15), "chunks": len(chunk_text(cleaned)),
    }
