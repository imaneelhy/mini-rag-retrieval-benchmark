from typing import List, Set
import numpy as np


def compute_recall_at_k(rankings: List[List[str]],
                        relevant_sets: List[Set[str]],
                        k: int) -> float:
    recalls = []
    for docs, rel in zip(rankings, relevant_sets):
        topk = docs[:k]
        hit = len(rel.intersection(topk)) > 0
        recalls.append(1.0 if hit else 0.0)
    return float(np.mean(recalls))


def compute_precision_at_k(rankings: List[List[str]],
                           relevant_sets: List[Set[str]],
                           k: int) -> float:
    precisions = []
    for docs, rel in zip(rankings, relevant_sets):
        topk = docs[:k]
        hits = len(rel.intersection(topk))
        precisions.append(hits / max(len(topk), 1))
    return float(np.mean(precisions))


def compute_mrr_at_k(rankings: List[List[str]],
                     relevant_sets: List[Set[str]],
                     k: int) -> float:
    mrrs = []
    for docs, rel in zip(rankings, relevant_sets):
        rr = 0.0
        for rank, doc_id in enumerate(docs[:k], start=1):
            if doc_id in rel:
                rr = 1.0 / rank
                break
        mrrs.append(rr)
    return float(np.mean(mrrs))
