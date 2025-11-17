
````markdown
# mini-rag-retrieval-benchmark

Mini benchmark comparing **BM25**, **dense retrieval**, and a **hybrid retriever**
for Retrieval-Augmented Generation (RAG) on a small synthetic domain corpus
(12 documents, 32 questions).

Instead of “just building a chatbot”, this project asks a research-style question:

> For a small domain knowledge base, how do BM25, dense embeddings, and a simple
> hybrid combination compare as retrievers for RAG-style question answering?

---

## Abstract

Retrieval-Augmented Generation (RAG) systems depend critically on the quality of
their retriever. In practice, engineers must choose between classical lexical
retrieval (e.g. BM25), neural dense retrieval, or a combination of both. In
this project, we construct a small synthetic corpus of 12 domain-specific
documents about RAG pipelines, retrieval methods, indexing, latency, security,
prompt engineering, and use cases (agriculture, finance, customer support). We
manually annotate 32 questions (easy/medium/hard) with single- and
multi-document relevance and benchmark three retrieval strategies: BM25, a dense
retriever based on `all-MiniLM-L6-v2`, and a simple hybrid that linearly
combines BM25 and dense scores.

We report Recall@k, Precision@k, and Mean Reciprocal Rank (MRR) on the
held-out question set, both overall and by difficulty. On this corpus, BM25
remains a very strong baseline (Recall@1 ≈ 0.91, MRR@3 ≈ 0.92), while dense
retrieval lags at rank 1 (Recall@1 ≈ 0.84) but achieves perfect Recall@5.
A hybrid retriever with α = 0.5, weighting BM25 and dense scores equally,
matches BM25’s Recall@1 and improves MRR@3 to ≈ 0.93. An α-sweep reveals a
clear sweet spot around α ≈ 0.5, and qualitative error analysis shows that
dense retrieval can recover semantically relevant documents for paraphrased or
multi-aspect questions where BM25 focuses on loosely related texts. Overall,
the results suggest that classical lexical retrieval remains competitive on
clean domain-specific queries, but combining lexical and semantic signals
yields benefits on more challenging questions.

---

## Repository structure

```text
mini-rag-retrieval-benchmark/
│
├─ src/
│   ├─ __init__.py
│   ├─ prepare_data.py       # create synthetic corpus + QA pairs
│   ├─ data_utils.py         # Document dataclass + loaders
│   ├─ retrievers.py         # BM25, Dense, Hybrid retrievers
│   ├─ metrics.py            # Recall@k, Precision@k, MRR@k
│   └─ benchmark.py          # runs experiment and saves metrics/plots
│
├─ notebooks/
│   └─ mini_rag_benchmark.ipynb   # Colab notebook (exploration & plots)
│
├─ requirements.txt
└─ README.md
````

---

## Dataset

The dataset is a **synthetic domain corpus** designed to resemble the kind of
documentation you’d see in real RAG projects.

**Documents**

* 12 documents in `data/docs/*.txt` (2–5 paragraphs each) covering:

  * Introduction to RAG
  * BM25 vs dense retrieval
  * RAG in agriculture
  * RAG for customer support
  * RAG in finance
  * System design for RAG pipelines
  * Indexing strategies (BM25, vector indexes, hybrid)
  * Latency & scaling
  * Security & privacy
  * Evaluation metrics (Recall@k, Precision@k, MRR)
  * Prompt engineering
  * Common failure modes (missed retrievals, hallucinations, partial answers)

**Questions**

* 32 questions in `data/qa_pairs.json`

  * `id`: question identifier
  * `question`: natural-language query
  * `difficulty`: `"easy" | "medium" | "hard"`
  * `relevant_docs`: list of 1–3 document IDs considered correct

Design:

* **Easy**: mostly keyword-style questions.
* **Medium**: paraphrased questions, but still focused on one document.
* **Hard**: more abstract wording, often requiring multiple documents
  (multi-doc relevance).

All data is created specifically for this project and can be extended or
replaced.

---

## Methods

We compare three retrieval strategies:

### 1. BM25 (lexical baseline)

A classical keyword-based retriever using `rank-bm25`.

* Tokenization: lowercase + whitespace split
* Scoring: BM25 over the full document text

### 2. Dense retrieval (semantic baseline)

A dense retriever implemented with [`sentence-transformers`](https://www.sbert.net/):

* Model: `all-MiniLM-L6-v2`
* Documents are encoded once and stored as vectors.
* Queries are encoded at runtime.
* Similarity: cosine similarity between query and document embeddings.

### 3. Hybrid retriever

A simple score-fusion model combining BM25 and dense similarities:

[
\text{hybrid}(d) = \alpha \cdot \tilde{s}*\text{BM25}(d)
+ (1 - \alpha) \cdot \tilde{s}*\text{dense}(d)
]

where (\tilde{s}) denotes min–max normalized scores across documents.

We sweep α ∈ {0.0, 0.25, 0.5, 0.75, 1.0}:

* α = 0 → pure dense retrieval
* α = 1 → pure BM25
* intermediate values → hybrid combinations

---

## Experimental setup

* **Corpus:** 12 synthetic RAG-related documents
* **Question set:** 32 labelled questions
* **Task:** document ranking for RAG

  * Input: question
  * Output: ranked list of document IDs
  * Ground truth: one or more relevant documents per question

**Metrics**

For each retrieval method, we compute:

* Recall@k (k ∈ {1, 3, 5})
* Precision@k (k ∈ {1, 3, 5})
* Mean Reciprocal Rank (MRR@k, k ∈ {1, 3, 5})

The main metrics discussed are:

* **Recall@1** – “How often is some relevant doc ranked first?”
* **MRR@3** – “On average, how high is the first relevant doc in the top-3?”

---

## Quantitative results

### Overall performance (32 questions)

Aggregated over all difficulties:

| Model          | Recall@1 | Recall@3 | Recall@5 | MRR@3 |
| -------------- | -------- | -------- | -------- | ----- |
| **BM25**       | 0.91     | 0.94     | 0.97     | 0.92  |
| **Dense**      | 0.84     | 0.97     | 1.00     | 0.91  |
| **Hybrid 0.5** | 0.91     | 0.97     | 1.00     | 0.93  |

Where “Hybrid 0.5” denotes α = 0.5 (equal weight on BM25 and dense scores).

These correspond to the JSON metrics:

* BM25:

  * Recall@1 = 0.9063, MRR@3 = 0.9219
* Dense:

  * Recall@1 = 0.8438, MRR@3 = 0.9063
* Hybrid (α = 0.5):

  * Recall@1 = 0.9063, MRR@3 = 0.9323

Full metrics, including precision values and Recall/MRR at different k, are
stored in `results/metrics.json`.

### By difficulty

Recall@1 / MRR@3 per difficulty:

* **Easy**

  * BM25: 0.88 / 0.94
  * Dense: 0.88 / 0.94
  * Hybrid: 0.88 / 0.92
* **Medium**

  * BM25: 0.90 / 0.90
  * Dense: 0.80 / 0.85
  * Hybrid: 0.90 / 0.90
* **Hard**

  * BM25: 0.93 / 0.93
  * Dense: 0.86 / 0.93
  * Hybrid: 0.93 / 0.96

All three methods behave similarly on **easy** questions.
On **medium** questions, BM25 and Hybrid outperform Dense at rank 1.
On **hard** questions, Hybrid achieves the highest MRR@3.

### Alpha sweep (hybrid)

We sweep α ∈ {0.0, 0.25, 0.5, 0.75, 1.0} and record:

* Recall@1
* MRR@3

The values (from `results/metrics.json`) are:

* Recall@1: [0.84, 0.88, 0.91, 0.91, 0.91]
* MRR@3:    [0.91, 0.92, 0.93, 0.92, 0.92]

The plot `results/alpha_sweep.png` shows:

* α = 0 (pure dense) → lowest Recall@1 and MRR@3.
* α ≈ 0.5 → **best MRR@3**.
* α ≥ 0.75 → performance similar to pure BM25.

This supports the idea that a balanced combination of lexical and semantic
signals is beneficial on this corpus.

---

## Discussion and error analysis

On this small synthetic corpus, **BM25 remains a very strong baseline**:
it ranks a relevant document first for about 91% of questions and achieves
MRR@3 ≈ 0.92. Dense retrieval underperforms at rank 1 (Recall@1 ≈ 0.84) but
recovers in deeper recall (Recall@5 = 1.0) and slightly higher Precision@3,
indicating that it surfaces many relevant documents within the top-5 even when
they are not always at position 1.

The hybrid retriever combines the strengths of both approaches. With α = 0.5,
it matches BM25’s Recall@1 while improving MRR@3 to ≈ 0.93 and achieving
Recall@5 = 1.0. The α-sweep reveals a clear sweet spot around α ≈ 0.5, where
hybrid MRR@3 is higher than both pure BM25 (α = 1) and pure Dense (α = 0).
This is consistent with the intuition that lexical signals (exact term overlap)
and dense signals (semantic similarity) are complementary.

Difficulty-wise, all methods perform similarly on easy, keyword-heavy queries.
On medium questions, BM25 and the hybrid model outperform dense retrieval at
rank 1, reflecting the importance of lexical overlap. On hard questions with
paraphrasing and multi-document relevance, the hybrid model achieves the best
MRR@3 (≈ 0.96), indicating that combining scores helps push at least one
relevant document higher in the ranking.

We also inspect individual queries. For example:

> “For a customer-support RAG system, which issues must be considered around
> latency, safety, and user trust?”

Relevant documents cover customer support, latency, and security. Dense
retrieval ranks the customer-support document first and surfaces latency and
security documents within the top results. BM25, in contrast, prefers a more
generic system-design document at rank 1 and only retrieves one relevant
document near the bottom of the top-5. This illustrates a typical case where
dense retrieval better captures the semantics of a complex query, while BM25
is dominated by surface-level keyword matches.

The main limitation of this benchmark is that the corpus and questions are
synthetic and relatively small. However, the setup is fully reproducible and
easy to extend, and it already exhibits realistic behaviours that mirror
design choices in practical RAG pipelines.

---

## How to run

Clone the repository and install dependencies:

```bash
git clone https://github.com/imaneelhy/mini-rag-retrieval-benchmark.git
cd mini-rag-retrieval-benchmark

pip install -r requirements.txt
```

Run the benchmark script:

```bash
python -m src.prepare_data    # optional: creates data/ if missing
python -m src.benchmark
```

This will:

1. Create the synthetic corpus under `data/` (if not already present).
2. Build BM25, Dense, and Hybrid retrievers.
3. Compute Recall@k, Precision@k, and MRR@k for each model.
4. Sweep α for the hybrid retriever.
5. Save:

   * `results/metrics.json` (all numbers),
   * `results/alpha_sweep.png` (alpha vs Recall@1 / MRR@3).

You can also open `notebooks/mini_rag_benchmark.ipynb` in Jupyter or Google
Colab to reproduce and extend the experiments interactively.

---

## Future work

Possible extensions of this mini-benchmark include:

* Adding more documents and questions from real technical blogs or public
  reports to increase realism.
* Trying alternative dense models (e.g. domain-specific sentence-transformers,
  E5, BGE, etc.).
* Evaluating additional hybrid strategies (e.g. rank fusion, cross-encoder
  reranking).
* Incorporating answer generation (LLM-in-the-loop) to evaluate full RAG
  rather than retrieval in isolation.
* Studying the impact of chunking strategies and document segmentation.

---

## License

This project is released under the [MIT License](LICENSE).

```
