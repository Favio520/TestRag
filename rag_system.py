from __future__ import annotations

import argparse
import json
import logging
from typing import List

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    INGEST_MODE_DEFAULT,
    NO_EVIDENCE_SCORE_THRESHOLD,
    RERANK_DEFAULT,
    SEARCH_TYPE_DEFAULT,
    TOP_K_DEFAULT,
)
from rag.ingest import ingest_documents, load_index_for_search
from rag.retrieval import retrieve_with_strategy

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def retrieve_top_fragments(
    query: str,
    data_dir: str = "data",
    index_dir: str = "faiss_index",
    top_k: int = TOP_K_DEFAULT,
    model_name: str = EMBEDDING_MODEL_NAME,
    search_type: str = SEARCH_TYPE_DEFAULT,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    document: str | None = None,
    folder: str | None = None,
    rerank: bool = RERANK_DEFAULT,
    score_threshold: float | None = NO_EVIDENCE_SCORE_THRESHOLD,
) -> List[Document]:
    if not query or not query.strip():
        raise ValueError("La consulta no puede estar vacia.")

    vector_store = load_index_for_search(
        data_dir=data_dir,
        index_dir=index_dir,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return retrieve_with_strategy(
        vector_store=vector_store,
        query=query.strip(),
        top_k=top_k,
        search_type=search_type,
        document=document,
        folder=folder,
        rerank=rerank,
        score_threshold=score_threshold,
    )


def main(
    query: str,
    data_dir: str = "data",
    index_dir: str = "faiss_index",
    top_k: int = TOP_K_DEFAULT,
    search_type: str = SEARCH_TYPE_DEFAULT,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    document: str | None = None,
    folder: str | None = None,
    rerank: bool = RERANK_DEFAULT,
    score_threshold: float | None = NO_EVIDENCE_SCORE_THRESHOLD,
) -> List[str]:
    docs = retrieve_top_fragments(
        query=query,
        data_dir=data_dir,
        index_dir=index_dir,
        top_k=top_k,
        search_type=search_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        document=document,
        folder=folder,
        rerank=rerank,
        score_threshold=score_threshold,
    )
    return [doc.page_content for doc in docs]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sistema RAG local con FAISS.")
    parser.add_argument(
        "query",
        nargs="*",
        help="Consulta del usuario (acepta texto con o sin comillas).",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Actualiza el indice FAISS desde data/ sin ejecutar una consulta.",
    )
    parser.add_argument(
        "--ingest-mode",
        default=INGEST_MODE_DEFAULT,
        choices=["incremental", "full"],
        help="Modo de ingesta: incremental agrega/corrige cambios puntuales; full reconstruye todo el indice.",
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
        "--model-name",
        default=EMBEDDING_MODEL_NAME,
        help="Modelo local de HuggingFace para embeddings.",
    )
    parser.add_argument(
        "--search-type",
        default=SEARCH_TYPE_DEFAULT,
        choices=["mmr", "similarity", "similarity_with_score", "hybrid"],
        help="Estrategia de recuperacion: mmr reduce redundancia; hybrid mezcla FAISS con una capa lexical.",
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
    parser.add_argument(
        "--document",
        default=None,
        help="Filtro opcional por documento. Acepta nombre de archivo, ruta relativa o titulo del documento.",
    )
    parser.add_argument(
        "--folder",
        default=None,
        help="Filtro opcional por carpeta o proyecto. Acepta una carpeta raiz o una ruta relativa parcial, por ejemplo BESSDailyreport o cliente/proyecto.",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Aplica una segunda pasada de reranking sobre los mejores candidatos recuperados.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=NO_EVIDENCE_SCORE_THRESHOLD,
        help="Umbral minimo de evidencia. Si el mejor resultado cae por debajo, se devuelve vacio.",
    )

    args = parser.parse_args()
    if args.ingest:
        result = ingest_documents(
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            model_name=args.model_name,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            mode=args.ingest_mode,
        )
        print("\nIngesta completada:\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        user_query = " ".join(args.query).strip() if args.query else input("Escribe tu consulta: ").strip()
        results = retrieve_top_fragments(
            query=user_query,
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            top_k=args.top_k,
            model_name=args.model_name,
            search_type=args.search_type,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            document=args.document,
            folder=args.folder,
            rerank=args.rerank,
            score_threshold=args.score_threshold,
        )

        print("\nTop fragmentos relevantes:\n")
        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source", "desconocido")
            chunk_id = doc.metadata.get("chunk_id", "n/a")
            print(f"[{i}] Fuente: {source} | Chunk: {chunk_id}")
            print(doc.page_content)
            print("-" * 80)
