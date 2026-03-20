from __future__ import annotations

import logging
import math
import re
from collections import Counter
from pathlib import PurePosixPath
from typing import Any, Callable, Iterable, List, Optional

from langchain_community.vectorstores import FAISS

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from .config import (
    HYBRID_FETCH_K_MULTIPLIER,
    HYBRID_LEXICAL_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    MMR_FETCH_K_MULTIPLIER,
    MMR_LAMBDA_MULT,
    NO_EVIDENCE_SCORE_THRESHOLD,
    RERANK_DEFAULT,
    RERANKER_MODEL_NAME,
    RERANK_TOP_N,
    SEARCH_TYPE_DEFAULT,
)

logger = logging.getLogger(__name__)
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
_RERANKER_CACHE: dict[str, Any] = {}


def normalize_document_value(value: Any) -> str:
    return str(value).strip().lower().replace("\\", "/")


def build_document_filter(document: Optional[str]) -> Optional[Callable[[dict[str, Any]], bool]]:
    normalized_document = normalize_document_value(document or "")
    if not normalized_document:
        return None

    requested_name = PurePosixPath(normalized_document).name
    requested_stem = PurePosixPath(requested_name).stem

    def metadata_matches(metadata: dict[str, Any]) -> bool:
        candidates: set[str] = set()
        for field in ("source_name", "relative_source", "source", "title"):
            raw_value = metadata.get(field)
            if not raw_value:
                continue

            normalized_value = normalize_document_value(raw_value)
            if not normalized_value:
                continue

            candidates.add(normalized_value)
            path_name = PurePosixPath(normalized_value).name
            if path_name:
                candidates.add(path_name)
                candidates.add(PurePosixPath(path_name).stem)

        return (
            normalized_document in candidates
            or requested_name in candidates
            or requested_stem in candidates
        )

    return metadata_matches


def document_key(doc: Document) -> str:
    metadata = doc.metadata
    return str(
        metadata.get("doc_id")
        or metadata.get("relative_source")
        or metadata.get("source")
        or metadata.get("source_name")
        or metadata.get("title")
        or id(doc)
    )


def tokenize_text(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def distance_to_relevance(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


def set_score_metadata(
    doc: Document,
    *,
    score: Optional[float] = None,
    raw_score: Optional[float] = None,
    vector_score: Optional[float] = None,
    lexical_score: Optional[float] = None,
    rerank_score: Optional[float] = None,
    score_type: Optional[str] = None,
) -> Document:
    if score is not None:
        doc.metadata["score"] = float(score)
    if raw_score is not None:
        doc.metadata["raw_score"] = float(raw_score)
    if vector_score is not None:
        doc.metadata["vector_score"] = float(vector_score)
    if lexical_score is not None:
        doc.metadata["lexical_score"] = float(lexical_score)
    if rerank_score is not None:
        doc.metadata["rerank_score"] = float(rerank_score)
    if score_type is not None:
        doc.metadata["score_type"] = score_type
    return doc


def get_all_documents(
    vector_store: FAISS,
    document_filter: Optional[Callable[[dict[str, Any]], bool]] = None,
) -> list[Document]:
    if not hasattr(vector_store, "index_to_docstore_id") or not hasattr(vector_store, "docstore"):
        return []

    documents: list[Document] = []
    for docstore_id in vector_store.index_to_docstore_id.values():
        doc = vector_store.docstore.search(docstore_id)
        if not isinstance(doc, Document):
            continue
        if document_filter is not None and not document_filter(doc.metadata):
            continue
        documents.append(doc)
    return documents


def get_vector_candidates(
    vector_store: FAISS,
    query: str,
    k: int,
    fetch_k: int,
    document_filter: Optional[Callable[[dict[str, Any]], bool]] = None,
) -> list[Document]:
    docs_with_score = vector_store.similarity_search_with_score(
        query,
        k=k,
        filter=document_filter,
        fetch_k=fetch_k,
    )
    results: list[Document] = []
    for doc, raw_score in docs_with_score:
        vector_score = distance_to_relevance(float(raw_score))
        results.append(
            set_score_metadata(
                doc,
                score=vector_score,
                raw_score=float(raw_score),
                vector_score=vector_score,
                score_type="vector_relevance",
            )
        )
    return results


def compute_lexical_candidates(query: str, docs: Iterable[Document]) -> list[Document]:
    query_tokens = tokenize_text(query)
    if not query_tokens:
        return []

    docs_list = list(docs)
    if not docs_list:
        return []

    query_terms = list(dict.fromkeys(query_tokens))
    tokenized_docs: list[list[str]] = []
    counters: list[Counter[str]] = []
    document_frequencies = Counter[str]()
    total_length = 0

    for doc in docs_list:
        tokens = tokenize_text(doc.page_content)
        tokenized_docs.append(tokens)
        counters.append(Counter(tokens))
        total_length += len(tokens)
        seen_terms = {term for term in query_terms if term in counters[-1]}
        document_frequencies.update(seen_terms)

    avg_doc_length = total_length / len(docs_list) if docs_list else 1.0
    k1 = 1.5
    b = 0.75
    scored_docs: list[tuple[Document, float]] = []

    for doc, tokens, counts in zip(docs_list, tokenized_docs, counters):
        if not tokens:
            continue
        doc_length = len(tokens)
        score = 0.0
        for term in query_terms:
            freq = counts.get(term, 0)
            if freq == 0:
                continue
            df = document_frequencies.get(term, 0)
            idf = math.log(1.0 + ((len(docs_list) - df + 0.5) / (df + 0.5)))
            numerator = freq * (k1 + 1.0)
            denominator = freq + k1 * (1.0 - b + b * (doc_length / max(avg_doc_length, 1.0)))
            score += idf * (numerator / denominator)
        if score > 0.0:
            scored_docs.append((doc, score))

    if not scored_docs:
        return []

    max_score = max(score for _, score in scored_docs) or 1.0
    results: list[Document] = []
    for doc, lexical_score in sorted(scored_docs, key=lambda item: item[1], reverse=True):
        results.append(
            set_score_metadata(
                doc,
                lexical_score=lexical_score / max_score,
            )
        )
    return results


def merge_hybrid_candidates(
    vector_candidates: list[Document],
    lexical_candidates: list[Document],
) -> list[Document]:
    merged: dict[str, Document] = {}

    for doc in vector_candidates:
        merged[document_key(doc)] = doc

    for doc in lexical_candidates:
        key = document_key(doc)
        existing = merged.get(key)
        if existing is None:
            merged[key] = doc
            continue
        lexical_score = doc.metadata.get("lexical_score")
        if lexical_score is not None:
            existing.metadata["lexical_score"] = float(lexical_score)

    results: list[Document] = []
    for doc in merged.values():
        vector_score = float(doc.metadata.get("vector_score", 0.0))
        lexical_score = float(doc.metadata.get("lexical_score", 0.0))
        hybrid_score = (vector_score * HYBRID_VECTOR_WEIGHT) + (lexical_score * HYBRID_LEXICAL_WEIGHT)
        results.append(
            set_score_metadata(
                doc,
                score=hybrid_score,
                vector_score=vector_score if vector_score > 0.0 else None,
                lexical_score=lexical_score if lexical_score > 0.0 else None,
                score_type="hybrid_relevance",
            )
        )
    return sorted(results, key=lambda doc: float(doc.metadata.get("score", 0.0)), reverse=True)


def get_reranker(model_name: str = RERANKER_MODEL_NAME) -> Any:
    cached = _RERANKER_CACHE.get(model_name)
    if cached is not None:
        return cached

    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(model_name)
    _RERANKER_CACHE[model_name] = reranker
    return reranker


def rerank_documents(query: str, docs: list[Document]) -> list[Document]:
    if len(docs) < 2:
        return docs

    top_docs = docs[: min(len(docs), RERANK_TOP_N)]
    remaining = docs[len(top_docs) :]

    try:
        reranker = get_reranker()
        pairs = [(query, doc.page_content) for doc in top_docs]
        scores = reranker.predict(pairs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo aplicar el reranker; se mantiene el orden original: %s", exc)
        return docs

    rescored_docs: list[Document] = []
    for doc, rerank_score in zip(top_docs, scores):
        rescored_docs.append(
            set_score_metadata(
                doc,
                rerank_score=float(rerank_score),
            )
        )

    rescored_docs.sort(key=lambda doc: float(doc.metadata.get("rerank_score", 0.0)), reverse=True)
    return rescored_docs + remaining


def passes_score_threshold(best_score: Optional[float], score_threshold: Optional[float]) -> bool:
    if best_score is None:
        return False
    if score_threshold is None or score_threshold <= 0:
        return True
    return best_score >= score_threshold


def retrieve_with_strategy(
    vector_store: FAISS,
    query: str,
    top_k: int,
    search_type: str = SEARCH_TYPE_DEFAULT,
    document: Optional[str] = None,
    rerank: bool = RERANK_DEFAULT,
    score_threshold: Optional[float] = NO_EVIDENCE_SCORE_THRESHOLD,
) -> List[Document]:
    normalized_search_type = search_type.strip().lower()
    document_filter = build_document_filter(document)
    result_k = max(top_k, RERANK_TOP_N) if rerank else top_k
    vector_fetch_k = max(result_k, result_k * MMR_FETCH_K_MULTIPLIER)

    if normalized_search_type == "hybrid":
        hybrid_fetch_k = max(result_k, result_k * HYBRID_FETCH_K_MULTIPLIER)
        vector_candidates = get_vector_candidates(
            vector_store,
            query=query,
            k=hybrid_fetch_k,
            fetch_k=hybrid_fetch_k,
            document_filter=document_filter,
        )
        lexical_candidates = compute_lexical_candidates(
            query,
            get_all_documents(vector_store, document_filter=document_filter),
        )[:hybrid_fetch_k]
        results = merge_hybrid_candidates(vector_candidates, lexical_candidates)
        best_score = float(results[0].metadata.get("score", 0.0)) if results else None
    elif normalized_search_type == "similarity":
        results = get_vector_candidates(
            vector_store,
            query=query,
            k=result_k,
            fetch_k=vector_fetch_k,
            document_filter=document_filter,
        )
        best_score = float(results[0].metadata.get("score", 0.0)) if results else None
    elif normalized_search_type == "similarity_with_score":
        results = get_vector_candidates(
            vector_store,
            query=query,
            k=result_k,
            fetch_k=vector_fetch_k,
            document_filter=document_filter,
        )
        best_score = float(results[0].metadata.get("score", 0.0)) if results else None
    else:
        vector_candidates = get_vector_candidates(
            vector_store,
            query=query,
            k=vector_fetch_k,
            fetch_k=vector_fetch_k,
            document_filter=document_filter,
        )
        best_score = float(vector_candidates[0].metadata.get("score", 0.0)) if vector_candidates else None
        vector_scores_by_key = {
            document_key(doc): {
                "score": doc.metadata.get("score"),
                "raw_score": doc.metadata.get("raw_score"),
                "vector_score": doc.metadata.get("vector_score"),
                "score_type": "vector_relevance",
            }
            for doc in vector_candidates
        }
        results = vector_store.max_marginal_relevance_search(
            query,
            k=result_k,
            fetch_k=vector_fetch_k,
            lambda_mult=MMR_LAMBDA_MULT,
            filter=document_filter,
        )
        for doc in results:
            score_payload = vector_scores_by_key.get(document_key(doc), {})
            set_score_metadata(
                doc,
                score=score_payload.get("score"),
                raw_score=score_payload.get("raw_score"),
                vector_score=score_payload.get("vector_score"),
                score_type=score_payload.get("score_type"),
            )

    if not passes_score_threshold(best_score, score_threshold):
        logger.info(
            "La consulta no supero el umbral minimo de evidencia (best_score=%s, threshold=%s).",
            best_score,
            score_threshold,
        )
        return []

    if rerank:
        results = rerank_documents(query, results)

    return results[:top_k]
