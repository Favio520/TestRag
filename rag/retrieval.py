from __future__ import annotations

from typing import List

from langchain_community.vectorstores import FAISS

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from .config import MMR_FETCH_K_MULTIPLIER, MMR_LAMBDA_MULT, SEARCH_TYPE_DEFAULT


def retrieve_with_strategy(
    vector_store: FAISS,
    query: str,
    top_k: int,
    search_type: str = SEARCH_TYPE_DEFAULT,
) -> List[Document]:
    normalized_search_type = search_type.strip().lower()

    if normalized_search_type == "similarity_with_score":
        docs_with_score = vector_store.similarity_search_with_score(query, k=top_k)
        results: List[Document] = []
        for doc, score in docs_with_score:
            doc.metadata["score"] = float(score)
            results.append(doc)
        return results

    if normalized_search_type == "similarity":
        return vector_store.similarity_search(query, k=top_k)

    fetch_k = max(top_k, top_k * MMR_FETCH_K_MULTIPLIER)
    return vector_store.max_marginal_relevance_search(
        query,
        k=top_k,
        fetch_k=fetch_k,
        lambda_mult=MMR_LAMBDA_MULT,
    )

