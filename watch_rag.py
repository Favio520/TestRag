from __future__ import annotations

import argparse
import logging

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL_NAME, INGEST_MODE_DEFAULT
from rag.watcher import RAGWatchService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watcher para ingesta automatica del RAG.")
    parser.add_argument("--data-dir", default="data", help="Carpeta de documentos a observar.")
    parser.add_argument("--index-dir", default="faiss_index", help="Carpeta del indice FAISS.")
    parser.add_argument("--debounce-seconds", type=int, default=10, help="Tiempo de espera antes de intentar la ingesta.")
    parser.add_argument("--stability-seconds", type=int, default=2, help="Tiempo para validar que el archivo dejo de cambiar.")
    parser.add_argument("--model-name", default=EMBEDDING_MODEL_NAME, help="Modelo de embeddings usado en la ingesta.")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Tamano de chunk usado en la ingesta.")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="Solapamiento entre chunks.")
    parser.add_argument(
        "--ingest-mode",
        default=INGEST_MODE_DEFAULT,
        choices=["incremental", "full"],
        help="Modo de ingesta disparado por el watcher.",
    )
    args = parser.parse_args()

    service = RAGWatchService(
        data_dir=args.data_dir,
        index_dir=args.index_dir,
        debounce_seconds=args.debounce_seconds,
        stability_seconds=args.stability_seconds,
        model_name=args.model_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        ingest_mode=args.ingest_mode,
    )
    service.start()
    service.wait_forever()
