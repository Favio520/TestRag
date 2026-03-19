from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL_NAME,
    INGEST_MODE_DEFAULT,
    SUPPORTED_EXTENSIONS,
)
from .ingest import ingest_documents

logger = logging.getLogger(__name__)


def _is_supported_path(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


class RAGWatchService:
    def __init__(
        self,
        data_dir: str = "data",
        index_dir: str = "faiss_index",
        debounce_seconds: int = 10,
        stability_seconds: int = 2,
        model_name: str = EMBEDDING_MODEL_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        ingest_mode: str = INGEST_MODE_DEFAULT,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.index_dir = index_dir
        self.debounce_seconds = debounce_seconds
        self.stability_seconds = stability_seconds
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ingest_mode = ingest_mode

        self._lock = threading.Lock()
        self._pending_paths: set[Path] = set()
        self._timer: Optional[threading.Timer] = None
        self._observer: Optional[Any] = None
        self._ingest_running = False

    def mark_changed(self, path: str | Path) -> None:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = (self.data_dir / candidate).resolve()
        else:
            candidate = candidate.resolve()

        if candidate.parent != self.data_dir and self.data_dir not in candidate.parents:
            return

        with self._lock:
            self._pending_paths.add(candidate)
            self._schedule_locked()

    def _schedule_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()

        self._timer = threading.Timer(self.debounce_seconds, self._process_pending)
        self._timer.daemon = True
        self._timer.start()

    def _process_pending(self) -> None:
        with self._lock:
            self._timer = None
            if self._ingest_running:
                self._schedule_locked()
                return

            pending_paths = set(self._pending_paths)
            self._ingest_running = True

        try:
            if not pending_paths:
                return

            pending_list = list(pending_paths)
            if not self._all_paths_stable(pending_list):
                logger.info("Se detectaron archivos aun inestables; se reprograma la ingesta.")
                with self._lock:
                    self._schedule_locked()
                return

            logger.info("Ejecutando watcher de ingesta incremental para %d cambios detectados.", len(pending_list))
            result = ingest_documents(
                data_dir=str(self.data_dir),
                index_dir=self.index_dir,
                model_name=self.model_name,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                mode=self.ingest_mode,
            )
            logger.info("Watcher de ingesta completado: %s", result)

            with self._lock:
                self._pending_paths.difference_update(pending_paths)
                if self._pending_paths:
                    self._schedule_locked()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error ejecutando watcher de ingesta: %s", exc)
            with self._lock:
                self._schedule_locked()
        finally:
            with self._lock:
                self._ingest_running = False

    def _all_paths_stable(self, paths: list[Path]) -> bool:
        first_snapshot = self._snapshot(paths)
        time.sleep(self.stability_seconds)
        second_snapshot = self._snapshot(paths)
        return first_snapshot == second_snapshot

    def _snapshot(self, paths: list[Path]) -> dict[str, tuple[bool, int, int]]:
        snapshot: dict[str, tuple[bool, int, int]] = {}
        for path in paths:
            if path.exists():
                stat = path.stat()
                snapshot[str(path)] = (True, stat.st_size, stat.st_mtime_ns)
            else:
                snapshot[str(path)] = (False, 0, 0)
        return snapshot

    def start(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"La carpeta observada no existe: {self.data_dir}")

        event_handler = RAGWatchHandler(self)
        observer = Observer()
        observer.schedule(event_handler, str(self.data_dir), recursive=True)
        observer.start()
        self._observer = observer
        logger.info(
            "Watcher RAG iniciado sobre %s con debounce de %ss.",
            self.data_dir,
            self.debounce_seconds,
        )

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        logger.info("Watcher RAG detenido.")

    def wait_forever(self) -> None:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()


class RAGWatchHandler(FileSystemEventHandler):
    def __init__(self, service: RAGWatchService) -> None:
        super().__init__()
        self.service = service

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def on_deleted(self, event: FileSystemEvent) -> None:
        self._handle_event(event)

    def on_moved(self, event: FileSystemEvent) -> None:
        self._handle_event(event)
        dest_path = getattr(event, "dest_path", None)
        if dest_path:
            self.service.mark_changed(dest_path)

    def _handle_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        src_path = Path(event.src_path)
        if src_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return

        self.service.mark_changed(src_path)
