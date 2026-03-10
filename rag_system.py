from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Callable, Iterator, List, Optional

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
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_DEFAULT = 3
BATCH_SIZE = 128
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md"}


def _iter_supported_files(data_dir: Path) -> Iterator[Path]:
    """Itera recursivamente los archivos soportados dentro de data_dir."""
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _extract_pdf_text(file_path: Path) -> str:
    text_parts: List[str] = []
    with fitz.open(file_path) as pdf_doc:
        for page in pdf_doc:
            page_text = page.get_text("text")
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts).strip()


def _extract_txt_text(file_path: Path) -> str:
    # Intento principal en utf-8 y fallback en latin-1 para archivos legacy.
    try:
        return file_path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1").strip()


def _extract_docx_text(file_path: Path) -> str:
    doc = DocxDocument(str(file_path))
    lines = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
    return "\n".join(lines).strip()


def _extract_md_text(file_path: Path) -> str:
    # Unstructured devuelve elementos semanticos; extraemos su texto.
    elements = partition_md(filename=str(file_path))
    lines = [el.text for el in elements if hasattr(el, "text") and el.text and el.text.strip()]
    return "\n".join(lines).strip()


EXTRACTORS: dict[str, Callable[[Path], str]] = {
    ".pdf": _extract_pdf_text,
    ".txt": _extract_txt_text,
    ".docx": _extract_docx_text,
    ".md": _extract_md_text,
}


def _safe_extract_text(file_path: Path) -> Optional[str]:
    """Extrae texto con manejo de excepciones y valida contenido no vacio."""
    extractor = EXTRACTORS.get(file_path.suffix.lower())
    if extractor is None:
        logger.warning("Formato no soportado, se omite: %s", file_path)
        return None

    try:
        text = extractor(file_path)
    except Exception as exc:  # noqa: BLE001 - robustez ante archivos corruptos
        logger.warning("No se pudo procesar %s: %s", file_path, exc)
        return None

    if not text or not text.strip():
        logger.warning("Archivo vacio o sin texto util, se omite: %s", file_path)
        return None
    return text.strip()


def _build_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Crea el modelo de embeddings local en CPU para limitar consumo de RAM/VRAM."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def _split_text_into_documents(
    text: str,
    source_path: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> List[Document]:
    chunks = splitter.split_text(text)
    return [
        Document(
            page_content=chunk,
            metadata={"source": str(source_path), "chunk_id": idx},
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
        text = _safe_extract_text(file_path)
        if text is None:
            skipped_files += 1
            continue

        documents = _split_text_into_documents(text, file_path, splitter)
        if not documents:
            logger.warning("No se pudieron generar chunks para %s", file_path)
            skipped_files += 1
            continue

        for batch in _batched(documents, batch_size):
            if vector_store is None:
                vector_store = FAISS.from_documents(batch, embeddings)
            else:
                vector_store.add_documents(batch)

        processed_files += 1
        total_chunks += len(documents)
        del text
        del documents
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
) -> FAISS:
    """Carga el indice si existe; si no, lo construye y lo guarda."""
    if not rebuild:
        loaded = load_faiss_index(index_dir=index_dir, model_name=model_name)
        if loaded is not None:
            logger.info("Indice FAISS cargado desde disco.")
            return loaded

    logger.info("Construyendo indice FAISS desde cero...")
    vector_store = build_faiss_index(data_dir=data_dir, model_name=model_name)
    save_faiss_index(vector_store, index_dir=index_dir)
    return vector_store


def retrieve_top_fragments(
    query: str,
    data_dir: str = "data",
    index_dir: str = "faiss_index",
    top_k: int = TOP_K_DEFAULT,
    rebuild: bool = False,
    model_name: str = EMBEDDING_MODEL_NAME,
) -> List[Document]:
    """Recupera los fragmentos mas relevantes para una consulta."""
    if not query or not query.strip():
        raise ValueError("La consulta no puede estar vacia.")

    vector_store = get_or_create_index(
        data_dir=data_dir,
        index_dir=index_dir,
        model_name=model_name,
        rebuild=rebuild,
    )
    return vector_store.similarity_search(query.strip(), k=top_k)


def main(
    query: str,
    data_dir: str = "data",
    index_dir: str = "faiss_index",
    top_k: int = TOP_K_DEFAULT,
    rebuild: bool = False,
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

    args = parser.parse_args()
    user_query = " ".join(args.query).strip() if args.query else input("Escribe tu consulta: ").strip()

    results = retrieve_top_fragments(
        query=user_query,
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        top_k=args.top_k,
        rebuild=args.rebuild,
        model_name=args.model_name,
    )

    print("\nTop fragmentos relevantes:\n")
    for i, doc in enumerate(results, start=1):
        source = doc.metadata.get("source", "desconocido")
        chunk_id = doc.metadata.get("chunk_id", "n/a")
        print(f"[{i}] Fuente: {source} | Chunk: {chunk_id}")
        print(doc.page_content)
        print("-" * 80)
