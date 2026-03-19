from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    MANIFEST_FILENAME,
    SUPPORTED_EXTENSIONS,
)

logger = logging.getLogger(__name__)


def build_base_metadata(file_path: Path) -> dict[str, Any]:
    return {
        "source": str(file_path),
        "source_name": file_path.name,
        "title": file_path.stem,
        "doc_type": file_path.suffix.lower().lstrip("."),
    }


def files_to_map(files: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(file_entry["path"]): file_entry for file_entry in files}


def manifest_state(manifest: Optional[dict[str, Any]]) -> dict[str, Any]:
    if manifest is None:
        return {}
    return {
        "data_dir": manifest.get("data_dir"),
        "model_name": manifest.get("model_name"),
        "chunk_size": manifest.get("chunk_size"),
        "chunk_overlap": manifest.get("chunk_overlap"),
        "supported_extensions": manifest.get("supported_extensions"),
        "files": manifest.get("files", []),
    }


def get_indexed_files_map(manifest: Optional[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("indexed_files", {}) if manifest else {}
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items() if isinstance(value, dict)}
    return {}


def iter_supported_files(data_dir: Path) -> Iterator[Path]:
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def build_data_manifest(
    data_dir: str = "data",
    model_name: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> dict[str, Any]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"La carpeta de datos no existe: {data_path.resolve()}")

    files = []
    for path in sorted(iter_supported_files(data_path)):
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
        "model_name": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "files": files,
    }


def manifest_path(index_dir: str = "faiss_index") -> Path:
    return Path(index_dir) / MANIFEST_FILENAME


def save_data_manifest(manifest: dict[str, Any], index_dir: str = "faiss_index") -> None:
    path = manifest_path(index_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_data_manifest(index_dir: str = "faiss_index") -> Optional[dict[str, Any]]:
    path = manifest_path(index_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("No se pudo leer el manifiesto del indice en %s: %s", path, exc)
        return None


def index_is_stale(
    data_dir: str = "data",
    index_dir: str = "faiss_index",
    model_name: str = EMBEDDING_MODEL_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> bool:
    saved_manifest = load_data_manifest(index_dir=index_dir)
    if saved_manifest is None:
        logger.info("No hay manifiesto del indice; se considera desactualizado.")
        return True

    current_manifest = build_data_manifest(
        data_dir=data_dir,
        model_name=model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return current_manifest != manifest_state(saved_manifest)


def relative_file_path(file_path: Path, data_dir: str) -> str:
    return str(file_path.relative_to(Path(data_dir)).as_posix())


def build_doc_id(relative_path: str, chunk_index: int) -> str:
    stable_prefix = hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"{stable_prefix}:{chunk_index}"

