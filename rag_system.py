from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional

import fitz  # PyMuPDF
from docx import Document as DocxDocument
from langchain_community.vectorstores import FAISS
from unstructured.partition.md import partition_md

try:
    from langchain_core.documents import Document
except ImportError:  # Compatibilidad con versiones antiguas
    from langchain.schema import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # Compatibilidad con versiones antiguas
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # Fallback si no existe langchain-huggingface
    from langchain_community.embeddings import HuggingFaceEmbeddings


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
TOP_K_DEFAULT = 4
BATCH_SIZE = 128
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}
MANIFEST_FILENAME = "data_manifest.json"
SEARCH_TYPE_DEFAULT = "similarity_with_score"
MMR_FETCH_K_MULTIPLIER = 4
MMR_LAMBDA_MULT = 0.7


def _build_base_metadata(file_path: Path) -> dict[str, Any]:
    return {
        "source": str(file_path),
        "source_name": file_path.name,
        "title": file_path.stem,
        "doc_type": file_path.suffix.lower().lstrip("."),
    }


def _read_text_file(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1")


def _split_markdown_sections(raw_text: str) -> List[dict[str, Any]]:
    sections: List[dict[str, Any]] = []
    current_heading = "Contenido general"
    current_lines: List[str] = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    sections.append({"text": text, "metadata": {"section": current_heading}})
                current_lines = []
            current_heading = stripped.lstrip("#").strip() or "Contenido general"
            continue

        current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append({"text": text, "metadata": {"section": current_heading}})

    return sections


def _iter_supported_files(data_dir: Path) -> Iterator[Path]:
    """Itera recursivamente los archivos soportados dentro de data_dir."""
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _build_data_manifest(data_dir: str = "data") -> dict:
    """
    Construye una firma liviana del contenido de data/ para detectar cambios
    sin reindexar en cada consulta.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"La carpeta de datos no existe: {data_path.resolve()}")

    files = []
    for path in sorted(_iter_supported_files(data_path)):
        stat = path.stat()
        files.append(
            {
                "path": str(path.relative_to(data_path).as_posix()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )

    return {
        "data_dir": str(data_path.resolve()),
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "files": files,
    }


def _manifest_path(index_dir: str = "faiss_index") -> Path:
    return Path(index_dir) / MANIFEST_FILENAME


def _save_data_manifest(manifest: dict, index_dir: str = "faiss_index") -> None:
    path = _manifest_path(index_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_data_manifest(index_dir: str = "faiss_index") -> Optional[dict]:
    path = _manifest_path(index_dir)
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("No se pudo leer el manifiesto del indice en %s: %s", path, exc)
        return None


def _index_is_stale(data_dir: str = "data", index_dir: str = "faiss_index") -> bool:
    saved_manifest = _load_data_manifest(index_dir=index_dir)
    if saved_manifest is None:
        logger.info("No hay manifiesto del indice; se reconstruira automaticamente.")
        return True

    current_manifest = _build_data_manifest(data_dir=data_dir)
    return current_manifest != saved_manifest


def _extract_pdf_segments(file_path: Path) -> List[dict[str, Any]]:
    segments: List[dict[str, Any]] = []
    with fitz.open(file_path) as pdf_doc:
        for page_index, page in enumerate(pdf_doc, start=1):
            page_text = page.get_text("text")
            if page_text:
                segments.append(
                    {
                        "text": page_text.strip(),
                        "metadata": {"page_start": page_index, "page_end": page_index},
                    }
                )
    return segments


def _extract_txt_segments(file_path: Path) -> List[dict[str, Any]]:
    text = _read_text_file(file_path).strip()
    return [{"text": text, "metadata": {}}] if text else []


def _extract_docx_segments(file_path: Path) -> List[dict[str, Any]]:
    doc = DocxDocument(str(file_path))
    sections: List[dict[str, Any]] = []
    current_heading = file_path.stem
    current_lines: List[str] = []

    for paragraph in doc.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue

        style_name = paragraph.style.name if paragraph.style is not None else ""
        if style_name.lower().startswith("heading"):
            if current_lines:
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append({"text": body, "metadata": {"section": current_heading}})
                current_lines = []
            current_heading = text
            continue

        current_lines.append(text)

    if current_lines:
        body = "\n".join(current_lines).strip()
        if body:
            sections.append({"text": body, "metadata": {"section": current_heading}})

    return sections


def _extract_md_segments(file_path: Path) -> List[dict[str, Any]]:
    raw_text = _read_text_file(file_path)
    sections = _split_markdown_sections(raw_text)
    if sections:
        return sections

    # Fallback a unstructured si el markdown no trae headings utiles.
    elements = partition_md(filename=str(file_path))
    lines = [el.text for el in elements if hasattr(el, "text") and el.text and el.text.strip()]
    text = "\n".join(lines).strip()
    return [{"text": text, "metadata": {}}] if text else []


EXTRACTORS: dict[str, Callable[[Path], List[dict[str, Any]]]] = {
    ".pdf": _extract_pdf_segments,
    ".txt": _extract_txt_segments,
    ".docx": _extract_docx_segments,
    ".md": _extract_md_segments,
}


def _safe_extract_segments(file_path: Path) -> Optional[List[dict[str, Any]]]:
    """Extrae segmentos con metadatos y valida contenido no vacio."""
    extractor = EXTRACTORS.get(file_path.suffix.lower())
    if extractor is None:
        logger.warning("Formato no soportado, se omite: %s", file_path)
        return None

    try:
        segments = extractor(file_path)
    except Exception as exc:  # noqa: BLE001 - robustez ante archivos corruptos
        logger.warning("No se pudo procesar %s: %s", file_path, exc)
        return None

    valid_segments = [
        segment
        for segment in segments
        if str(segment.get("text", "")).strip()
    ]
    if not valid_segments:
        logger.warning("Archivo vacio o sin texto util, se omite: %s", file_path)
        return None
    return valid_segments


def _build_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Crea el modelo de embeddings local en CPU para limitar consumo de RAM/VRAM."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _split_text_into_documents(
    text: str,
    metadata: dict[str, Any],
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    chunks = splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={**metadata, "chunk_id": idx},
        )
        for idx, chunk in enumerate(chunks)
        if chunk.strip()
    ]


def _batched(items: List[Document], batch_size: int) -> Iterator[List[Document]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def build_faiss_index(
    data_dir: str = "data",
    model_name: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    batch_size: int = BATCH_SIZE,
) -> FAISS:
    """
    Construye un indice FAISS desde documentos locales.
    Usa procesamiento por archivo y lotes para evitar picos de memoria.
    """
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
    embeddings = _build_embeddings(model_name)

    vector_store: Optional[FAISS] = None
    processed_files = 0
    skipped_files = 0
    total_chunks = 0

    for file_path in _iter_supported_files(data_path):
        segments = _safe_extract_segments(file_path)
        if segments is None:
            skipped_files += 1
            continue

        file_documents: List[Document] = []
        base_metadata = _build_base_metadata(file_path)
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            segment_metadata = {**base_metadata, **dict(segment.get("metadata", {}))}
            segment_documents = _split_text_into_documents(text, segment_metadata, splitter)
            file_documents.extend(segment_documents)

        if not file_documents:
            logger.warning("No se pudieron generar chunks para %s", file_path)
            skipped_files += 1
            continue

        for batch in _batched(file_documents, batch_size):
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)

        processed_files += 1
        total_chunks += len(file_documents)
        del file_documents
        gc.collect()

    if vector_store is None:
        raise RuntimeError(
            "No se construyo el indice FAISS: no hubo archivos validos en data/."
        )

    logger.info(
        "Indice generado. Archivos procesados: %d | Omitidos: %d | Chunks: %d",
        processed_files,
        skipped_files,
        total_chunks,
    )
    return vector_store


def save_faiss_index(vector_store: FAISS, index_dir: str = "faiss_index") -> None:
    """Guarda el indice FAISS en disco para evitar reprocesar documentos."""
    path = Path(index_dir)
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))
    logger.info("Indice FAISS guardado en: %s", path.resolve())


def load_faiss_index(
    index_dir: str = "faiss_index",
    model_name: str = EMBEDDING_MODEL_NAME,
) -> Optional[FAISS]:
    """Carga un indice FAISS previamente guardado, si existe."""
    path = Path(index_dir)
    if not path.exists():
        return None

    if not (path / "index.faiss").exists() or not (path / "index.pkl").exists():
        return None

    embeddings = _build_embeddings(model_name)
    return FAISS.load_local(
        str(path),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_or_create_index(
    data_dir: str = "data",
    index_dir: str = "faiss_index",
    model_name: str = EMBEDDING_MODEL_NAME,
    rebuild: bool = False,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> FAISS:
    """Carga el indice si existe; si no, lo construye y lo guarda."""
    if not rebuild:
        loaded = load_faiss_index(index_dir=index_dir, model_name=model_name)
        if loaded is not None:
            if _index_is_stale(data_dir=data_dir, index_dir=index_dir):
                logger.info("Se detectaron cambios en data/; se reconstruira el indice.")
            else:
                logger.info("Indice FAISS cargado desde disco.")
                return loaded

    logger.info("Construyendo indice FAISS desde cero...")
    vector_store = build_faiss_index(
        data_dir=data_dir,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    save_faiss_index(vector_store, index_dir=index_dir)
    _save_data_manifest(_build_data_manifest(data_dir=data_dir), index_dir=index_dir)
    return vector_store


def _retrieve_with_strategy(
    vector_store: FAISS,
    query: str,
    top_k: int,
    search_type: str = SEARCH_TYPE_DEFAULT,
) -> List[Document]:
    """
    Ejecuta la recuperacion usando una estrategia configurable.
    MMR suele devolver resultados menos repetidos que una similitud plana.
    """
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


def retrieve_top_fragments(
    query: str,
    data_dir: str = "data",
    index_dir: str = "faiss_index",
    top_k: int = TOP_K_DEFAULT,
    rebuild: bool = False,
    model_name: str = EMBEDDING_MODEL_NAME,
    search_type: str = SEARCH_TYPE_DEFAULT,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Recupera los fragmentos mas relevantes para una consulta."""
    if not query or not query.strip():
        raise ValueError("La consulta no puede estar vacia.")

    vector_store = get_or_create_index(
        data_dir=data_dir,
        index_dir=index_dir,
        model_name=model_name,
        rebuild=rebuild,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return _retrieve_with_strategy(
        vector_store=vector_store,
        query=query.strip(),
        top_k=top_k,
        search_type=search_type,
    )


def main(
    query: str,
    data_dir: str = "data",
    index_dir: str = "faiss_index",
    top_k: int = TOP_K_DEFAULT,
    rebuild: bool = False,
    search_type: str = SEARCH_TYPE_DEFAULT,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Funcion principal solicitada:
    recibe una consulta y devuelve los 3 fragmentos mas relevantes.
    """
    docs = retrieve_top_fragments(
        query=query,
        data_dir=data_dir,
        index_dir=index_dir,
        top_k=top_k,
        rebuild=rebuild,
        search_type=search_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return [doc.page_content for doc in docs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema RAG local con FAISS.")
    parser.add_argument(
        "query",
        nargs="*",
        help="Consulta del usuario (acepta texto con o sin comillas).",
    )
    parser.add_argument("--data-dir", default="data", help="Carpeta con documentos fuente.")
    parser.add_argument("--index-dir", default="faiss_index", help="Carpeta del indice FAISS.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K_DEFAULT,
        help="Cantidad de fragmentos a devolver.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruye el indice ignorando el cache en disco.",
    )
    parser.add_argument(
        "--model-name",
        default=EMBEDDING_MODEL_NAME,
        help="Modelo local de HuggingFace para embeddings.",
    )
    parser.add_argument(
        "--search-type",
        default=SEARCH_TYPE_DEFAULT,
        choices=["mmr", "similarity", "similarity_with_score"],
        help="Estrategia de recuperacion: mmr reduce redundancia entre fragmentos.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Tamano de chunk para indexacion. Manuales tecnicos suelen rendir mejor con chunks mas grandes.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=CHUNK_OVERLAP,
        help="Solapamiento entre chunks para preservar continuidad entre secciones.",
    )

    args = parser.parse_args()
    user_query = " ".join(args.query).strip() if args.query else input("Escribe tu consulta: ").strip()

    results = retrieve_top_fragments(
        query=user_query,
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        top_k=args.top_k,
        rebuild=args.rebuild,
        model_name=args.model_name,
        search_type=args.search_type,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print("\nTop fragmentos relevantes:\n")
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "desconocido")
        chunk_id = doc.metadata.get("chunk_id", "n/a")
        print(f"[{i}] Fuente: {source} | Chunk: {chunk_id}")
        print(doc.page_content)
        print("-" * 80)
