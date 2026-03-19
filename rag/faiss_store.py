from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Iterator, List, Optional

from langchain_community.vectorstores import FAISS

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import BATCH_SIZE, CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME
from .document_loaders import safe_extract_segments
from .metadata import build_base_metadata, build_doc_id, iter_supported_files, relative_file_path


def build_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_text_into_documents(
    text: str,
    metadata: dict[str, Any],
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    chunks = splitter.split_text(text)
    return [
        Document(page_content=chunk, metadata={**metadata, "chunk_id": idx})
        for idx, chunk in enumerate(chunks)
        if chunk.strip()
    ]


def batched(items: List[Document], batch_size: int) -> Iterator[List[Document]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def prepare_documents_for_file(
    file_path: Path,
    data_dir: str,
    splitter: RecursiveCharacterTextSplitter,
) -> tuple[List[Document], List[str]]:
    segments = safe_extract_segments(file_path)
    if segments is None:
        return [], []

    rel_path = relative_file_path(file_path, data_dir)
    documents: List[Document] = []
    doc_ids: List[str] = []
    base_metadata = build_base_metadata(file_path)

    for segment in segments:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        segment_metadata = {
            **base_metadata,
            "relative_source": rel_path,
            **dict(segment.get("metadata", {})),
        }
        documents.extend(split_text_into_documents(text, segment_metadata, splitter))

    for idx in range(len(documents)):
        doc_ids.append(build_doc_id(rel_path, idx))

    for idx, document in enumerate(documents):
        document.metadata["doc_id"] = doc_ids[idx]

    return documents, doc_ids


def build_faiss_index(
    data_dir: str = "data",
    model_name: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    batch_size: int = BATCH_SIZE,
) -> tuple[FAISS, dict[str, dict[str, Any]]]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"La carpeta de datos no existe: {data_path.resolve()}")
    if chunk_size < 100:
        raise ValueError("chunk_size debe ser mayor o igual a 100.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap no puede ser negativo.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap debe ser menor que chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    embeddings = build_embeddings(model_name)

    vector_store: Optional[FAISS] = None
    indexed_files: dict[str, dict[str, Any]] = {}
    total_chunks = 0

    for file_path in iter_supported_files(data_path):
        rel_path = relative_file_path(file_path, data_dir)
        file_documents, doc_ids = prepare_documents_for_file(file_path, data_dir, splitter)
        if not file_documents:
            continue

        for batch_index, batch in enumerate(batched(file_documents, batch_size)):
            start = batch_index * batch_size
            batch_ids = doc_ids[start : start + len(batch)]
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings, ids=batch_ids)
            else:
                vector_store.add_documents(batch, ids=batch_ids)

        indexed_files[rel_path] = {
            "doc_ids": doc_ids,
            "chunk_count": len(file_documents),
        }
        total_chunks += len(file_documents)
        del file_documents
        gc.collect()

    if vector_store is None:
        raise RuntimeError(
            "No se construyo el indice FAISS: no hubo archivos validos en data/."
        )

    return vector_store, indexed_files


def save_faiss_index(vector_store: FAISS, index_dir: str = "faiss_index") -> None:
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))


def load_faiss_index(
    index_dir: str = "faiss_index",
    model_name: str = EMBEDDING_MODEL_NAME,
) -> Optional[FAISS]:
    path = Path(index_dir)
    if not path.exists():
        return None
    if not (path / "index.faiss").exists() or not (path / "index.pkl").exists():
        return None
    embeddings = build_embeddings(model_name)
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )

