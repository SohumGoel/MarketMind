"""
RAG pipeline for SEC filings and long-form financial documents.

Workflow:
    1. Chunk document into overlapping text windows
    2. Embed chunks with SentenceBERT (or similar)
    3. Index into FAISS vector store
    4. Query: retrieve top-k relevant passages for a given query

TODO (Sohum): Implement chunking, embedding, indexing, and retrieval.

Reference query template:
    "material risk factors, revenue guidance, and earnings surprise"
"""


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for financial documents.

    TODO (Sohum): Implement this class.
    """

    def index_document(self, document_text: str, doc_id: str) -> None:
        """Chunk, embed, and index a document."""
        raise NotImplementedError("RAGPipeline.index_document not yet implemented.")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Retrieve top-k relevant passages for a query.

        Returns:
            List of dicts with keys: 'text', 'doc_id', 'score'
        """
        raise NotImplementedError("RAGPipeline.retrieve not yet implemented.")
