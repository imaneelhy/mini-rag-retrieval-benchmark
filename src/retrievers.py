from typing import List
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def simple_tokenize(text: str) -> List[str]:
    return text.lower().split()


class BM25Retriever:
    """BM25 keyword-based retriever using rank-bm25."""

    def __init__(self, corpus: List[str], doc_ids: List[str]):
        self.doc_ids = doc_ids
        self.tokenized_corpus = [simple_tokenize(doc) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, k: int = 5):
        tokenized_query = simple_tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(self.doc_ids, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]


class DenseRetriever:
    """Dense embedding-based retriever using sentence-transformers."""

    def __init__(self, corpus: List[str], doc_ids: List[str],
                 model_name: str = "all-MiniLM-L6-v2"):
        self.doc_ids = doc_ids
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings = self.model.encode(
            corpus, convert_to_numpy=True, show_progress_bar=True
        )

    def search(self, query: str, k: int = 5):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_emb, self.doc_embeddings)[0]
        ranked = sorted(
            zip(self.doc_ids, sims),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]

    def scores_for_all(self, query: str) -> np.ndarray:
        query_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_emb, self.doc_embeddings)[0]
        return sims


class HybridRetriever:
    """
    Simple hybrid retriever combining BM25 and dense similarities.

    hybrid_score = alpha * bm25_norm + (1 - alpha) * dense_norm
    """

    def __init__(self, corpus: List[str], doc_ids: List[str], dense_retriever: DenseRetriever):
        self.doc_ids = doc_ids
        self.bm25 = BM25Retriever(corpus, doc_ids)
        self.dense = dense_retriever

    def search(self, query: str, k: int = 5, alpha: float = 0.5):
        tokenized_query = simple_tokenize(query)
        bm25_scores = self.bm25.bm25.get_scores(tokenized_query)
        dense_scores = self.dense.scores_for_all(query)

        def norm(scores):
            scores = scores.astype("float32")
            if scores.max() - scores.min() < 1e-8:
                return np.zeros_like(scores)
            return (scores - scores.min()) / (scores.max() - scores.min())

        bm25_norm = norm(bm25_scores)
        dense_norm = norm(dense_scores)
        hybrid = alpha * bm25_norm + (1.0 - alpha) * dense_norm

        ranked = sorted(
            zip(self.doc_ids, hybrid),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:k]
