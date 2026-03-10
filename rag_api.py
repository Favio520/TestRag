from __future__ import annotations

import argparse
import logging
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag_system import TOP_K_DEFAULT, retrieve_top_fragments


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
    rebuild: bool = False
    data_dir: str = "data"
    index_dir: str = "faiss_index"


class SearchResult(BaseModel):
    content: str
    source: str
    chunk_id: int | str


class SearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[SearchResult]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "rag-api",
        "status": "ok",
        "health": "/health",
        "search": "POST /search",
    }


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    try:
        docs = retrieve_top_fragments(
            query=request.query,
            data_dir=request.data_dir,
            index_dir=request.index_dir,
            top_k=request.top_k,
            rebuild=request.rebuild,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - API robusta
        logger.exception("Error en recuperacion RAG")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results = [
        SearchResult(
            content=doc.page_content,
            source=str(doc.metadata.get("source", "desconocido")),
            chunk_id=doc.metadata.get("chunk_id", "n/a"),
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
