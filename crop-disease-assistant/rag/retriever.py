"""
retriever.py
------------
Queries the ChromaDB vector store using a two-stage strategy:
  1. Metadata pre-filter by crop + disease
  2. Semantic similarity ranking within that filtered set

Returns the top-k context chunks as a single joined string
ready to be inserted into the LLM prompt.
"""

import os
import chromadb
from chromadb.utils import embedding_functions

CHROMA_DIR  = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION  = "crop_disease_knowledge"
EMBED_MODEL = "intfloat/multilingual-e5-large"  # must match ingest.py

_client     = None
_collection = None


def _get_collection():
    """Lazy-load ChromaDB collection (singleton)."""
    global _client, _collection
    if _collection is None:
        if not os.path.exists(CHROMA_DIR):
            raise FileNotFoundError(
                f"ChromaDB index not found at {CHROMA_DIR}. "
                "Run rag/ingest.py first to build the index."
            )
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        _client     = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_collection(
            name=COLLECTION,
            embedding_function=ef,
        )
        print(f"[retriever] Loaded ChromaDB collection: {_collection.count()} chunks")
    return _collection


def retrieve(context: dict, top_k: int = 4) -> str:
    """
    Retrieve relevant document chunks for a given context dict
    (output of context_builder.build_context).

    Strategy
    --------
    1. Try metadata-filtered query: where crop AND disease match
    2. If fewer than 2 results, fall back to crop-only filter
    3. If still < 2, fall back to unfiltered semantic search
    4. Returns top_k chunks joined as a single context string

    Parameters
    ----------
    context : dict   Output from context_builder.build_context()
    top_k   : int    Number of chunks to return (default 4)

    Returns
    -------
    str   Retrieved context for LLM prompt, or empty string if nothing found.
    """
    collection     = _get_collection()
    query          = context["retrieval_query"]
    crop           = context["crop"].lower()
    disease        = context["disease"].lower()
    low_confidence = context["low_confidence"]

    # If low confidence, broaden to crop-level only
    if low_confidence:
        results = _query_with_filter(
            collection, query,
            {"$and": [{"crop": {"$eq": crop}}, {"disease": {"$eq": disease}}]},
            top_k,
        )
        if not results:
            results = _query_unfiltered(collection, query, top_k)
        return _format_results(results)

    # Stage 1: crop + disease filter
    results = _query_with_filter(
        collection, query,
        {"$and": [{"crop": {"$eq": crop}}, {"disease": {"$eq": disease}}]},
        top_k,
    )

    # Stage 2: crop-only fallback
    if len(results) < 2:
        print(f"[retriever] Stage 1 returned {len(results)} results. Falling back to crop filter.")
        results = _query_with_filter(collection, query, {"crop": {"$eq": crop}}, top_k)

    # Stage 3: unfiltered fallback
    if len(results) < 2:
        print(f"[retriever] Stage 2 returned {len(results)} results. Falling back to unfiltered.")
        results = _query_unfiltered(collection, query, top_k)

    return _format_results(results)

# def retrieve(context: dict, top_k: int = 4) -> str:
#     """
#     Retrieve relevant context chunks from ChromaDB based on the input context.
#     Uses a two-stage retrieval strategy:
#         1. Filter by crop + disease metadata, then rank by similarity
#         2. If too few results, filter by crop only, then rank
#         3. If still too few, do an unfiltered similarity search
#     """
#     collection     = _get_collection()
#     query          = context["retrieval_query"]
#     crop           = context["crop"].lower()
#     disease        = context["disease"].lower()

#     # Stage 1: crop + disease filter
#     results = _query_with_filter(collection, query, {"crop": crop, "disease": disease}, top_k)

#     # Stage 2: crop-only fallback
#     if len(results) < 2:
#         print(f"[retriever] Stage 1 returned {len(results)} results. Falling back to crop filter.")
#         results = _query_with_filter(collection, query, {"crop": crop}, top_k)

#     # Stage 3: unfiltered fallback
#     if len(results) < 2:
#         print(f"[retriever] Stage 2 returned {len(results)} results. Falling back to unfiltered.")
#         results = _query_unfiltered(collection, query, top_k)

#     return _format_results(results)


def _query_with_filter(collection, query: str, where: dict, top_k: int) -> list[str]:
    """Query ChromaDB with a metadata filter."""
    try:
        res = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0]
        return [d for d in docs if d and len(d.strip()) > 50]
    except Exception as e:
        print(f"[retriever] Filter query failed: {e}")
        return []


def _query_unfiltered(collection, query: str, top_k: int) -> list[str]:
    """Fallback: unfiltered semantic search."""
    try:
        res = collection.query(
            query_texts=[query],
            n_results=min(top_k, collection.count()),
            include=["documents"],
        )
        docs = res.get("documents", [[]])[0]
        return [d for d in docs if d and len(d.strip()) > 50]
    except Exception as e:
        print(f"[retriever] Unfiltered query failed: {e}")
        return []


def _format_results(chunks: list[str]) -> str:
    """Join chunks into a single context string for the LLM."""
    if not chunks:
        return ""
    return "\n\n---\n\n".join(chunks)
