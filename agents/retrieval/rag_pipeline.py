"""
RAG pipeline for SEC filings and long-form financial documents.

Uses SentenceBERT for embeddings and FAISS for fast similarity search.
"""

import io
import logging
import os
import sys
import numpy as np
import faiss
from dotenv import load_dotenv

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

load_dotenv()
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    _old_stdout, sys.stdout = sys.stdout, io.StringIO()
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
    finally:
        sys.stdout = _old_stdout

from sentence_transformers import SentenceTransformer


_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_CHUNK_SIZE    = 512
_CHUNK_OVERLAP = 64

_RAG_QUERY = (
    "material risk factors, revenue guidance, earnings surprise, "
    "forward outlook, capital expenditure, debt obligations"
)


def _chunk_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


class RAGPipeline:
    """
    Chunk → embed → index SEC documents, then retrieve top-k passages
    relevant to financial forecasting.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        import sys, io
        _old_stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            self.model = SentenceTransformer(model_name)
        finally:
            sys.stderr = _old_stderr
        self.chunks: list[str] = []
        self.doc_ids: list[str] = []
        self.index: faiss.IndexFlatIP = None

    def index_document(self, document_text: str, doc_id: str) -> None:
        new_chunks = _chunk_text(document_text)
        if not new_chunks:
            return

        embeddings = self.model.encode(new_chunks, normalize_embeddings=True, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        if self.index is None:
            self.index = faiss.IndexFlatIP(embeddings.shape[1])

        self.index.add(embeddings)
        self.chunks.extend(new_chunks)
        self.doc_ids.extend([doc_id] * len(new_chunks))

    def retrieve(self, query: str = _RAG_QUERY, top_k: int = 5) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []

        q_emb = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        q_emb = np.array(q_emb, dtype="float32")

        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(q_emb, top_k)

        return [
            {
                "text": self.chunks[idx],
                "doc_id": self.doc_ids[idx],
                "score": float(scores[0][rank]),
            }
            for rank, idx in enumerate(indices[0])
            if idx < len(self.chunks)
        ]

    def reset(self) -> None:
        self.chunks = []
        self.doc_ids = []
        self.index = None
