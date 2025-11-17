from dataclasses import dataclass
from typing import List, Dict, Tuple
import os
import json


@dataclass
class Document:
    doc_id: str
    text: str


def load_documents(docs_dir: str = "data/docs") -> List[Document]:
    """Load all .txt files from docs_dir as Document objects."""
    if not os.path.exists(docs_dir):
        raise FileNotFoundError(f"{docs_dir} does not exist. "
                                "Run src/prepare_data.py first.")
    documents: List[Document] = []
    for fname in os.listdir(docs_dir):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(docs_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        doc_id = os.path.splitext(fname)[0]
        documents.append(Document(doc_id=doc_id, text=text))
    documents.sort(key=lambda d: d.doc_id)
    return documents


def load_qa_pairs(path: str = "data/qa_pairs.json") -> List[Dict]:
    """Load QA pairs from JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist. "
                                "Run src/prepare_data.py first.")
    with open(path, "r", encoding="utf-8") as f:
        qa_list = json.load(f)
    return qa_list


def build_corpus_and_ids(documents: List[Document]) -> Tuple[List[str], List[str]]:
    """Return parallel lists of document texts and doc_ids."""
    corpus = [d.text for d in documents]
    doc_ids = [d.doc_id for d in documents]
    return corpus, doc_ids
