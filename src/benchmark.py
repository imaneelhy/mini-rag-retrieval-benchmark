import os
import json
from typing import List, Set, Dict

import matplotlib.pyplot as plt

from .prepare_data import create_synthetic_corpus
from .data_utils import load_documents, load_qa_pairs, build_corpus_and_ids
from .retrievers import DenseRetriever, BM25Retriever, HybridRetriever
from .metrics import compute_recall_at_k, compute_precision_at_k, compute_mrr_at_k


def summarize_metrics(name: str, rankings: List[List[str]], relevant_sets: List[Set[str]]):
    print(f"=== {name} ===")
    for k in [1, 3, 5]:
        r = compute_recall_at_k(rankings, relevant_sets, k)
        p = compute_precision_at_k(rankings, relevant_sets, k)
        mrr = compute_mrr_at_k(rankings, relevant_sets, k)
        print(f"k={k}: Recall={r:.3f}, Precision={p:.3f}, MRR={mrr:.3f}")


def run_benchmark(alpha: float = 0.5) -> Dict:
    # Ensure data exists
    if not os.path.exists("data/qa_pairs.json") or not os.path.exists("data/docs"):
        create_synthetic_corpus("data")

    documents = load_documents("data/docs")
    qa_pairs = load_qa_pairs("data/qa_pairs.json")
    corpus, doc_ids = build_corpus_and_ids(documents)

    questions = [qa["question"] for qa in qa_pairs]
    relevant_sets = [set(qa["relevant_docs"]) for qa in qa_pairs]
    difficulties = [qa["difficulty"] for qa in qa_pairs]

    print(f"Number of documents: {len(documents)}")
    print(f"Number of QA pairs: {len(qa_pairs)}")

    dense = DenseRetriever(corpus, doc_ids, model_name="all-MiniLM-L6-v2")
    bm25 = BM25Retriever(corpus, doc_ids)
    hybrid = HybridRetriever(corpus, doc_ids, dense_retriever=dense)

    rankings_bm25, rankings_dense, rankings_hybrid = [], [], []

    for q in questions:
        bm25_res = bm25.search(q, k=10)
        dense_res = dense.search(q, k=10)
        hybrid_res = hybrid.search(q, k=10, alpha=alpha)

        rankings_bm25.append([d for d, _ in bm25_res])
        rankings_dense.append([d for d, _ in dense_res])
        rankings_hybrid.append([d for d, _ in hybrid_res])

    summarize_metrics("BM25", rankings_bm25, relevant_sets)
    summarize_metrics("Dense", rankings_dense, relevant_sets)
    summarize_metrics(f"Hybrid (alpha={alpha})", rankings_hybrid, relevant_sets)

    # Difficulty-wise metrics (printed only)
    def filter_by_diff(ranks, rels, diffs, target):
        idxs = [i for i, d in enumerate(diffs) if d == target]
        return [ranks[i] for i in idxs], [rels[i] for i in idxs]

    for diff in ["easy", "medium", "hard"]:
        print(f"\n--- Difficulty: {diff} ---")
        for name, ranks in [("BM25", rankings_bm25),
                            ("Dense", rankings_dense),
                            ("Hybrid", rankings_hybrid)]:
            r_sub, rel_sub = filter_by_diff(ranks, relevant_sets, difficulties, diff)
            if not r_sub:
                continue
            r1 = compute_recall_at_k(r_sub, rel_sub, 1)
            mrr3 = compute_mrr_at_k(r_sub, rel_sub, 3)
            print(f"{name}: Recall@1={r1:.3f}, MRR@3={mrr3:.3f}")

    # Alpha sweep
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    recall_alpha, mrr_alpha = [], []

    for a in alphas:
        rankings_h = []
        for q in questions:
            res = hybrid.search(q, k=10, alpha=a)
            rankings_h.append([d for d, _ in res])
        recall_alpha.append(compute_recall_at_k(rankings_h, relevant_sets, 1))
        mrr_alpha.append(compute_mrr_at_k(rankings_h, relevant_sets, 3))

    os.makedirs("results", exist_ok=True)

    # Plot alpha sweep
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(alphas, recall_alpha, marker="o")
    plt.xlabel("alpha (BM25 weight)")
    plt.ylabel("Recall@1")
    plt.title("Hybrid Recall@1 vs alpha")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(alphas, mrr_alpha, marker="o")
    plt.xlabel("alpha (BM25 weight)")
    plt.ylabel("MRR@3")
    plt.title("Hybrid MRR@3 vs alpha")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("results/alpha_sweep.png", dpi=200)
    plt.close()

    # Metrics dict (very close to what you printed in Colab)
    metrics = {}
    for name, ranks in [
        ("bm25", rankings_bm25),
        ("dense", rankings_dense),
        ("hybrid_alpha0.5", rankings_hybrid),
    ]:
        metrics[name] = {}
        for k in [1, 3, 5]:
            metrics[name][f"recall@{k}"] = compute_recall_at_k(ranks, relevant_sets, k)
            metrics[name][f"precision@{k}"] = compute_precision_at_k(ranks, relevant_sets, k)
            metrics[name][f"mrr@{k}"] = compute_mrr_at_k(ranks, relevant_sets, k)

    metrics["hybrid_alpha_sweep"] = {
        "alphas": alphas,
        "recall@1": recall_alpha,
        "mrr@3": mrr_alpha,
    }

    with open("results/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    run_benchmark(alpha=0.5)
