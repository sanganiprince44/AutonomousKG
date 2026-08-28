"""Transparent extraction metrics for a manually labelled test set."""
from __future__ import annotations


def score(predicted: list[dict], expected: list[dict], fields: tuple[str, ...]) -> dict:
    make_key = lambda item: tuple(str(item.get(field, "")).strip().casefold() for field in fields)
    pred, gold = {make_key(item) for item in predicted}, {make_key(item) for item in expected}
    tp, fp, fn = len(pred & gold), len(pred - gold), len(gold - pred)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0}
