from __future__ import annotations

import argparse
import logging
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_system import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    NO_EVIDENCE_SCORE_THRESHOLD,
    RERANK_DEFAULT,
    SEARCH_TYPE_DEFAULT,
    TOP_K_DEFAULT,
    ingest_documents,
    retrieve_top_fragments,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="RAG Local API",
    description="API HTTP para recuperar fragmentos relevantes desde FAISS.",
    version="1.0.0",
)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Consulta del usuario.")
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=10)
    search_type: str = Field(default=SEARCH_TYPE_DEFAULT, pattern="^(mmr|similarity|similarity_with_score|hybrid)$")
    document: str | None = Field(default=None, description="Filtro opcional por documento.")
    folder: str | None = Field(default=None, description="Filtro opcional por carpeta o proyecto.")
    rerank: bool = RERANK_DEFAULT
    score_threshold: float | None = Field(default=NO_EVIDENCE_SCORE_THRESHOLD, ge=0.0, le=1.0)
    model_name: str = EMBEDDING_MODEL_NAME
    chunk_size: int = Field(default=CHUNK_SIZE, ge=100, le=4000)
    chunk_overlap: int = Field(default=CHUNK_OVERLAP, ge=0, le=1000)
    data_dir: str = "data"
    index_dir: str = "faiss_index"


class IngestRequest(BaseModel):
    data_dir: str = "data"
    index_dir: str = "faiss_index"
    model_name: str = EMBEDDING_MODEL_NAME
    chunk_size: int = Field(default=CHUNK_SIZE, ge=100, le=4000)
    chunk_overlap: int = Field(default=CHUNK_OVERLAP, ge=0, le=1000)
    mode: str = Field(default="incremental", pattern="^(incremental|full)$")


class SearchResult(BaseModel):
    content: str
    source: str
    relative_source: str | None = None
    chunk_id: int | str
    source_name: str | None = None
    title: str | None = None
    doc_type: str | None = None
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    score: float | None = None
    raw_score: float | None = None
    vector_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None
    score_type: str | None = None


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchResult]


class IngestResponse(BaseModel):
    status: str
    mode: str
    data_dir: str
    index_dir: str
    model_name: str
    chunk_size: int
    chunk_overlap: int
    documents: int
    added_files: int
    updated_files: int
    deleted_files: int


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "rag-api",
        "status": "ok",
        "health": "/health",
        "ingest": "POST /ingest",
        "search": "POST /search",
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    try:
        result = ingest_documents(
            data_dir=request.data_dir,
            index_dir=request.index_dir,
            model_name=request.model_name,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            mode=request.mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - API robusta
        logger.exception("Error en ingesta RAG")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(**result)


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        docs = retrieve_top_fragments(
            query=request.query,
            data_dir=request.data_dir,
            index_dir=request.index_dir,
            top_k=request.top_k,
            model_name=request.model_name,
            search_type=request.search_type,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            document=request.document,
            folder=request.folder,
            rerank=request.rerank,
            score_threshold=request.score_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - API robusta
        logger.exception("Error en recuperacion RAG")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results = [
        SearchResult(
            content=doc.page_content,
            source=str(doc.metadata.get("source", "desconocido")),
            relative_source=doc.metadata.get("relative_source"),
            chunk_id=doc.metadata.get("chunk_id", "n/a"),
            source_name=doc.metadata.get("source_name"),
            title=doc.metadata.get("title"),
            doc_type=doc.metadata.get("doc_type"),
            section=doc.metadata.get("section"),
            page_start=doc.metadata.get("page_start"),
            page_end=doc.metadata.get("page_end"),
            score=doc.metadata.get("score"),
            raw_score=doc.metadata.get("raw_score"),
            vector_score=doc.metadata.get("vector_score"),
            lexical_score=doc.metadata.get("lexical_score"),
            rerank_score=doc.metadata.get("rerank_score"),
            score_type=doc.metadata.get("score_type"),
        )
        for doc in docs
    ]

    return SearchResponse(
        query=request.query,
        top_k=request.top_k,
        results=results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Levanta la API local del sistema RAG.")
    parser.add_argument("--host", default="127.0.0.1", help="Host de escucha.")
    parser.add_argument("--port", type=int, default=8000, help="Puerto HTTP.")
    args = parser.parse_args()

    uvicorn.run("rag_api:app", host=args.host, port=args.port, reload=False)
