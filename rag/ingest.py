from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME, INGEST_MODE_DEFAULT
from .faiss_store import build_faiss_index, load_faiss_index, prepare_documents_for_file, save_faiss_index
from .metadata import (
    build_data_manifest,
    files_to_map,
    get_indexed_files_map,
    index_is_stale,
    load_data_manifest,
    manifest_state,
    save_data_manifest,
)

logger = logging.getLogger(__name__)


def ingest_documents(
    data_dir: str = "data",
    index_dir: str = "faiss_index",
    model_name: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    mode: str = INGEST_MODE_DEFAULT,
) -> dict[str, Any]:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"full", "incremental"}:
        raise ValueError("mode debe ser 'full' o 'incremental'.")

    current_manifest = build_data_manifest(
        data_dir=data_dir,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if normalized_mode == "full":
        logger.info("Construyendo indice FAISS completo desde cero...")
        vector_store, indexed_files = build_faiss_index(
            data_dir=data_dir,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        save_faiss_index(vector_store, index_dir=index_dir)
        manifest = {**current_manifest, "indexed_files": indexed_files}
        save_data_manifest(manifest, index_dir=index_dir)
        return {
            "status": "indexed",
            "mode": "full",
            "data_dir": str(Path(data_dir).resolve()),
            "index_dir": str(Path(index_dir).resolve()),
            "model_name": model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "documents": len(current_manifest["files"]),
            "added_files": len(current_manifest["files"]),
            "updated_files": 0,
            "deleted_files": 0,
        }

    existing_manifest = load_data_manifest(index_dir=index_dir)
    if existing_manifest is None:
        logger.info("No existe un indice previo; se realizara una ingesta completa inicial.")
        return ingest_documents(
            data_dir=data_dir,
            index_dir=index_dir,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            mode="full",
        )

    saved_state = manifest_state(existing_manifest)
    if (
        saved_state.get("data_dir") != current_manifest["data_dir"]
        or saved_state.get("model_name") != current_manifest["model_name"]
        or saved_state.get("chunk_size") != current_manifest["chunk_size"]
        or saved_state.get("chunk_overlap") != current_manifest["chunk_overlap"]
        or saved_state.get("supported_extensions") != current_manifest["supported_extensions"]
    ):
        raise RuntimeError(
            "La configuracion de indexacion cambio. Ejecuta una ingesta completa para mantener el indice consistente."
        )

    existing_files = files_to_map(saved_state.get("files", []))
    current_files = files_to_map(current_manifest["files"])
    indexed_files = get_indexed_files_map(existing_manifest)

    added_paths = [path for path in current_files if path not in existing_files]
    modified_paths = [
        path
        for path, file_entry in current_files.items()
        if path in existing_files and file_entry != existing_files[path]
    ]
    deleted_paths = [path for path in existing_files if path not in current_files]

    if not added_paths and not modified_paths and not deleted_paths:
        logger.info("No hay cambios para ingesta incremental.")
        return {
            "status": "indexed",
            "mode": "incremental",
            "data_dir": str(Path(data_dir).resolve()),
            "index_dir": str(Path(index_dir).resolve()),
            "model_name": model_name,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "documents": len(current_manifest["files"]),
            "added_files": 0,
            "updated_files": 0,
            "deleted_files": 0,
        }

    unresolved_paths = [
        path for path in modified_paths + deleted_paths if not indexed_files.get(path, {}).get("doc_ids")
    ]
    if unresolved_paths:
        raise RuntimeError(
            "No se puede actualizar incrementalmente algunos archivos ya existentes porque el manifiesto anterior no guarda ids por documento. Ejecuta una ingesta completa una vez y luego podras seguir con incremental."
        )

    vector_store = load_faiss_index(index_dir=index_dir, model_name=model_name)
    if vector_store is None:
        logger.info("No se pudo cargar el indice previo; se realizara una ingesta completa inicial.")
        return ingest_documents(
            data_dir=data_dir,
            index_dir=index_dir,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            mode="full",
        )

    delete_ids: list[str] = []
    for path in modified_paths + deleted_paths:
        delete_ids.extend([str(doc_id) for doc_id in indexed_files[path].get("doc_ids", [])])

    if delete_ids:
        vector_store.delete(ids=delete_ids)
        for path in modified_paths + deleted_paths:
            indexed_files.pop(path, None)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    added_count = 0
    updated_count = 0
    for path in added_paths + modified_paths:
        file_path = Path(data_dir) / path
        documents, doc_ids = prepare_documents_for_file(file_path, data_dir, splitter)
        if not documents:
            continue
        vector_store.add_documents(documents, ids=doc_ids)
        indexed_files[path] = {
            "doc_ids": doc_ids,
            "chunk_count": len(documents),
        }
        if path in added_paths:
            added_count += 1
        else:
            updated_count += 1

    save_faiss_index(vector_store, index_dir=index_dir)
    manifest = {**current_manifest, "indexed_files": indexed_files}
    save_data_manifest(manifest, index_dir=index_dir)
    return {
        "status": "indexed",
        "mode": "incremental",
        "data_dir": str(Path(data_dir).resolve()),
        "index_dir": str(Path(index_dir).resolve()),
        "model_name": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "documents": len(current_manifest["files"]),
        "added_files": added_count,
        "updated_files": updated_count,
        "deleted_files": len(deleted_paths),
    }


def load_index_for_search(
    data_dir: str = "data",
    index_dir: str = "faiss_index",
    model_name: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    loaded = load_faiss_index(index_dir=index_dir, model_name=model_name)
    if loaded is None:
        raise RuntimeError(
            "El indice FAISS no existe. Ejecuta una ingesta antes de consultar."
        )

    if index_is_stale(
        data_dir=data_dir,
        index_dir=index_dir,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    ):
        raise RuntimeError(
            "El indice FAISS esta desactualizado respecto a data/ o a la configuracion de indexacion. Ejecuta una ingesta antes de consultar."
        )

    logger.info("Indice FAISS cargado desde disco para consulta.")
    return loaded
