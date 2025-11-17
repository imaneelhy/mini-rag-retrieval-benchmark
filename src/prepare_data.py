import os
import json


def create_synthetic_corpus(base_dir: str = "data") -> None:
    """
    Create a small synthetic RAG corpus:
    - 12 documents in base_dir/docs
    - 32 QA pairs in base_dir/qa_pairs.json
    """
    docs_dir = os.path.join(base_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    docs = {
        # (same content as in your notebook; shortened comments here)
        "doc1_intro_to_rag": """
Title: Introduction to Retrieval-Augmented Generation

Retrieval-Augmented Generation (RAG) is an approach where a generative language
model is connected to an external knowledge base. At query time, a retriever
selects the most relevant documents for the user question. These documents are
then provided as additional context to the generator, which produces an answer
grounded in the retrieved evidence.

RAG is particularly useful when the knowledge base changes frequently, or when
the model must be able to answer questions about niche domains that were not
fully covered during pre-training. Instead of re-training the entire model,
we simply update or extend the underlying document collection.
""",
        "doc2_bm25_vs_dense": """
Title: BM25 and Dense Retrieval Methods

BM25 is a classical ranking function widely used in information retrieval. It
relies on term frequency and inverse document frequency, and works directly on
tokenized text without any learned embeddings. BM25 is simple, robust, and
remains highly competitive for many keyword-style search tasks.

Dense retrieval methods, on the other hand, represent queries and documents as
vectors in a shared embedding space. These embeddings are typically produced by
a neural network, such as a transformer encoder, and similarity is measured
with dot product or cosine similarity. Dense retrievers can capture semantic
similarity beyond exact keyword overlap, but they usually require more compute
and a training phase.
""",
        "doc3_rag_in_agriculture": """
Title: Applications of RAG in Agricultural Knowledge Systems

In agricultural settings, domain experts need quick access to heterogeneous
information sources such as technical reports, regulations, weather bulletins,
and internal best-practice documents. Retrieval-Augmented Generation (RAG) can
be used to build chatbots that answer questions about crop diseases, fertilizer
dosages, or trade regulations by retrieving relevant passages from a curated
corpus of documents.

Compared to a pure retrieval system, a RAG-based assistant can synthesize
information from multiple documents and generate coherent natural language
answers. This reduces the time experts spend searching manually and improves
access to strategic and technical knowledge within the organization.
""",
        "doc4_system_design": """
Title: System Design Considerations for RAG Pipelines

A practical RAG system typically consists of three layers: ingestion, retrieval,
and generation. Ingestion is responsible for parsing raw documents (PDF, HTML,
text) and storing them in a normalized format. The retrieval layer indexes the
documents, using BM25, dense embeddings, or a hybrid of both. The generation
layer is usually a large language model that receives both the query and the
retrieved passages.

When designing such systems, engineers must consider latency, index update
frequency, data privacy constraints, and evaluation methodology. In many cases,
a hybrid retriever that combines BM25 and dense retrieval provides a good
balance between robustness and semantic understanding.
""",
        "doc5_evaluation_metrics": """
Title: Evaluation Metrics for Information Retrieval and RAG

To evaluate retrieval quality, practitioners often rely on metrics such as
Recall@k, Precision@k, and Mean Reciprocal Rank (MRR). Recall@k measures how
often at least one relevant document appears in the top-k retrieved results,
while Precision@k quantifies the fraction of retrieved items that are relevant.

MRR focuses specifically on the rank of the first relevant document, assigning
higher scores when relevant documents are ranked closer to the top. For RAG
systems, offline evaluation of retrievers can be combined with human judgments
of generated answers to obtain a more complete view of system quality.
""",
        "doc6_latency_scaling": """
Title: Latency and Scaling Considerations

RAG systems deployed in production must satisfy strict latency constraints,
especially when integrated into interactive chat interfaces. Users typically
expect a response within a few hundred milliseconds, which limits the number
of documents that can be retrieved and the size of the language model that can
be used for generation.

To scale RAG to large document collections, engineers adopt techniques such as
sharding, approximate nearest neighbor search, and request batching. Caching
popular queries and precomputing representations can further reduce response
times.
""",
        "doc7_security_privacy": """
Title: Security and Privacy in RAG Systems

RAG pipelines often operate over sensitive corporate documents. As a result,
access control and audit logging are critical. The retriever must respect
document-level permissions so that users cannot see content they are not
authorized to access. Moreover, logs of queries and retrieved passages need to
be stored securely and anonymized whenever possible.

From a privacy perspective, it is important to prevent the language model from
memorizing or leaking confidential information. Techniques such as redaction,
on-the-fly de-identification, and private deployment of models can mitigate
some of these risks.
""",
        "doc8_prompt_engineering": """
Title: Prompt Engineering for RAG

Even with a strong retriever, the quality of RAG outputs depends heavily on
prompt design. A typical RAG prompt contains the user question, the retrieved
passages, and explicit instructions such as: 'answer based only on the context'
or 'cite sources in your response'. Carefully crafted prompts can reduce
hallucinations and encourage the model to ask for clarification when the
retrieved evidence is insufficient.

Prompt engineering also involves decisions about how many passages to show, how
they are formatted, and whether the system should generate chain-of-thought
explanations or concise answers.
""",
        "doc9_rag_customer_support": """
Title: RAG for Customer Support

In customer support scenarios, RAG systems can answer questions about product
features, troubleshooting steps, and company policies by retrieving from an
up-to-date knowledge base. Compared to static FAQ pages, a RAG-powered virtual
assistant can handle more diverse queries and adapt to new documents as soon as
they are ingested.

However, support environments are sensitive to incorrect answers. As a result,
organizations often combine RAG with confidence estimation, fallback rules, and
human-in-the-loop review for high-risk requests.
""",
        "doc10_rag_finance": """
Title: RAG in Financial Applications

Financial institutions are exploring RAG to help analysts navigate long
regulatory documents, research reports, and internal risk assessments. In this
domain, strict compliance requirements demand traceable answers and explicit
links back to source documents. The retrieval component must be carefully tuned
to avoid missing critical passages that could change the interpretation of
regulations or risk guidelines.

Latency constraints can also be less strict for research workflows, which
allows the use of larger models and more retrieved documents, but increases
compute costs.
""",
        "doc11_indexing_strategies": """
Title: Indexing Strategies for Large-Scale Corpora

Indexing strategies determine how documents are stored and accessed by the
retriever. BM25 typically relies on inverted indexes, which map terms to the
documents that contain them. Dense retrieval uses vector indexes such as FAISS
or ScaNN, which support approximate nearest neighbor search in high-dimensional
spaces.

Hybrid systems may maintain both an inverted index and a vector index, merging
scores at query time. The choice of index has strong implications for memory
usage, latency, and update frequency.
""",
        "doc12_failure_modes": """
Title: Common Failure Modes of RAG Systems

RAG systems can fail in several characteristic ways. The retriever may miss
critical documents due to vocabulary mismatch or outdated indexes. Even when
relevant passages are retrieved, the generator might ignore them and hallucinate
facts, especially when prompts are poorly designed or when the model has strong
prior beliefs.

Another failure mode involves partial answers: the system returns only a subset
of the necessary information, which can be dangerous in domains such as
medicine or law. Robust evaluation and monitoring are therefore essential to
detect and mitigate these issues.
"""
    }

    for doc_id, text in docs.items():
        with open(os.path.join(docs_dir, f"{doc_id}.txt"), "w", encoding="utf-8") as f:
            f.write(text)

    # 32 QA pairs exactly as in your notebook:
    qa_pairs = [
        # EASY
        {"id": "q1", "difficulty": "easy",
         "question": "What is Retrieval-Augmented Generation (RAG)?",
         "relevant_docs": ["doc1_intro_to_rag"]},
        {"id": "q2", "difficulty": "easy",
         "question": "What does BM25 rely on to rank documents?",
         "relevant_docs": ["doc2_bm25_vs_dense"]},
        {"id": "q3", "difficulty": "easy",
         "question": "Which layer parses raw PDFs and HTML in a RAG pipeline?",
         "relevant_docs": ["doc4_system_design"]},
        {"id": "q4", "difficulty": "easy",
         "question": "Which metric focuses on the rank of the first relevant document?",
         "relevant_docs": ["doc5_evaluation_metrics"]},
        {"id": "q5", "difficulty": "easy",
         "question": "Which component must respect document-level permissions in RAG?",
         "relevant_docs": ["doc7_security_privacy"]},
        {"id": "q6", "difficulty": "easy",
         "question": "In which domain can RAG assist analysts reading regulations and research reports?",
         "relevant_docs": ["doc10_rag_finance"]},
        {"id": "q7", "difficulty": "easy",
         "question": "Which document discusses using RAG for crop disease and fertilizer questions?",
         "relevant_docs": ["doc3_rag_in_agriculture"]},
        {"id": "q8", "difficulty": "easy",
         "question": "Which document describes RAG for answering customer questions about products and policies?",
         "relevant_docs": ["doc9_rag_customer_support"]},

        # MEDIUM
        {"id": "q9", "difficulty": "medium",
         "question": "Why can RAG be preferable to retraining a language model when new documents arrive?",
         "relevant_docs": ["doc1_intro_to_rag"]},
        {"id": "q10", "difficulty": "medium",
         "question": "Which retrieval approach can capture semantic similarity beyond exact keyword overlap?",
         "relevant_docs": ["doc2_bm25_vs_dense"]},
        {"id": "q11", "difficulty": "medium",
         "question": "How can RAG-based assistants reduce the time agricultural experts spend manually searching for information?",
         "relevant_docs": ["doc3_rag_in_agriculture"]},
        {"id": "q12", "difficulty": "medium",
         "question": "What are the three typical layers of a practical RAG system?",
         "relevant_docs": ["doc4_system_design"]},
        {"id": "q13", "difficulty": "medium",
         "question": "Which metrics could you use to report how often relevant documents appear near the top of the ranking?",
         "relevant_docs": ["doc5_evaluation_metrics"]},
        {"id": "q14", "difficulty": "medium",
         "question": "What techniques can help scale RAG to very large document collections while keeping latency under control?",
         "relevant_docs": ["doc6_latency_scaling"]},
        {"id": "q15", "difficulty": "medium",
         "question": "How can organizations prevent RAG systems from leaking confidential information?",
         "relevant_docs": ["doc7_security_privacy"]},
        {"id": "q16", "difficulty": "medium",
         "question": "Why is it important to instruct the model to answer only based on retrieved context in the prompt?",
         "relevant_docs": ["doc8_prompt_engineering"]},
        {"id": "q17", "difficulty": "medium",
         "question": "What additional safeguards might customer support teams add around a RAG assistant to avoid harmful responses?",
         "relevant_docs": ["doc9_rag_customer_support"]},
        {"id": "q18", "difficulty": "medium",
         "question": "Why are traceable answers with explicit links to source documents especially important in finance?",
         "relevant_docs": ["doc10_rag_finance"]},

        # HARD
        {"id": "q19", "difficulty": "hard",
         "question": "Which methods go beyond simple term matching and instead compare continuous representations of queries and documents?",
         "relevant_docs": ["doc2_bm25_vs_dense", "doc11_indexing_strategies"]},
        {"id": "q20", "difficulty": "hard",
         "question": "When building a large-scale RAG index, which kinds of data structures might you use for lexical and vector-based retrieval?",
         "relevant_docs": ["doc11_indexing_strategies"]},
        {"id": "q21", "difficulty": "hard",
         "question": "For agricultural question-answering, which modules of a RAG system must be engineered to handle heterogeneous reports and regulations?",
         "relevant_docs": ["doc3_rag_in_agriculture", "doc4_system_design"]},
        {"id": "q22", "difficulty": "hard",
         "question": "In a chat interface where users expect sub-second responses, which aspects of the RAG pipeline become bottlenecks?",
         "relevant_docs": ["doc4_system_design", "doc6_latency_scaling"]},
        {"id": "q23", "difficulty": "hard",
         "question": "What types of failures can occur when the retriever misses key passages or the generator ignores retrieved evidence?",
         "relevant_docs": ["doc12_failure_modes"]},
        {"id": "q24", "difficulty": "hard",
         "question": "Which documents describe problems where the model gives only partial answers or hallucinates unsupported statements?",
         "relevant_docs": ["doc12_failure_modes", "doc8_prompt_engineering"]},
        {"id": "q25", "difficulty": "hard",
         "question": "In sensitive domains like finance and internal corporate data, which concerns arise around access control and compliance for RAG?",
         "relevant_docs": ["doc7_security_privacy", "doc10_rag_finance"]},
        {"id": "q26", "difficulty": "hard",
         "question": "If you wanted to evaluate not just whether relevant documents are retrieved but also how the generator uses them, which documents discuss such evaluation ideas?",
         "relevant_docs": ["doc5_evaluation_metrics", "doc12_failure_modes"]},
        {"id": "q27", "difficulty": "hard",
         "question": "What combination of retrieval approaches and indexing strategies could provide a good trade-off between robustness and semantic understanding?",
         "relevant_docs": ["doc2_bm25_vs_dense", "doc11_indexing_strategies", "doc4_system_design"]},
        {"id": "q28", "difficulty": "hard",
         "question": "For a customer-support RAG system, which issues must be considered around latency, safety, and user trust?",
         "relevant_docs": ["doc6_latency_scaling", "doc7_security_privacy", "doc9_rag_customer_support"]},
        {"id": "q29", "difficulty": "hard",
         "question": "Which parts of the RAG pipeline are most likely to cause users to lose trust if they fail silently?",
         "relevant_docs": ["doc4_system_design", "doc12_failure_modes"]},
        {"id": "q30", "difficulty": "hard",
         "question": "Across the described use cases, which texts emphasize the importance of human oversight or human-in-the-loop processes?",
         "relevant_docs": ["doc9_rag_customer_support", "doc10_rag_finance", "doc12_failure_modes"]},
        {"id": "q31", "difficulty": "hard",
         "question": "Where can you find a discussion of how many passages to show and how to format them in the prompt?",
         "relevant_docs": ["doc8_prompt_engineering"]},
        {"id": "q32", "difficulty": "hard",
         "question": "Which documents describe trade-offs between latency constraints and the ability to use larger models or more retrieved passages?",
         "relevant_docs": ["doc6_latency_scaling", "doc10_rag_finance"]}
    ]

    qa_path = os.path.join(base_dir, "qa_pairs.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Created {len(docs)} documents and {len(qa_pairs)} QA pairs in {base_dir}/")


if __name__ == "__main__":
    create_synthetic_corpus()
