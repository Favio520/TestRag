#!/bin/bash

# Configuración de variables
PY=".venv/bin/python3"
PIP=".venv/bin/pip"
SCRIPT="rag_system.py"
API_SCRIPT="rag_api.py"
WATCH_SCRIPT="watch_rag.py"
DATA_DIR="data"
INDEX_DIR="faiss_index"
TOP_K="4"
API_HOST="127.0.0.1"
API_PORT="8000"

# Función para mostrar uso (equivalente a :usage)
usage() {
    echo "Uso: ./run.sh [comando]"
    echo "Comandos disponibles:"
    echo "  install   - Instala las dependencias"
    echo "  ingest    - Actualiza el indice FAISS de forma incremental"
    echo "  ingest-full - Reconstruye el indice FAISS completo"
    echo "  watch     - Observa data/ y dispara ingesta incremental automatica"
    echo "  ask       - Realiza una consulta (ej: ./run.sh ask \"mi pregunta\")"
    echo "  api       - Inicia el servidor API"
    echo "  help      - Muestra este mensaje"
    exit 1
}

# Verificar si existe el entorno virtual
if [ ! -f "$PY" ]; then
    echo "[ERROR] No se encontró el entorno virtual en \".venv\"."
    echo "Crea uno con: python3 -m venv .venv"
    exit 1
fi

# Lógica de comandos
case "$1" in
    install)
        if [ ! -f "requirements.txt" ]; then
            echo "[ERROR] No se encontró requirements.txt"
            exit 1
        fi
        $PIP install -r requirements.txt
        ;;

    ingest)
        $PY "$SCRIPT" --ingest --ingest-mode incremental --data-dir "$DATA_DIR" --index-dir "$INDEX_DIR" --chunk-size 900 --chunk-overlap 180
        ;;

    ingest-full)
        $PY "$SCRIPT" --ingest --ingest-mode full --data-dir "$DATA_DIR" --index-dir "$INDEX_DIR" --chunk-size 900 --chunk-overlap 180
        ;;

    watch)
        $PY "$WATCH_SCRIPT" --data-dir "$DATA_DIR" --index-dir "$INDEX_DIR" --debounce-seconds 10 --stability-seconds 2 --ingest-mode incremental
        ;;

    ask)
        shift
        if [ -z "$1" ]; then
            echo "[ERROR] Debes enviar una consulta."
            exit 1
        fi
        $PY "$SCRIPT" "$*" --data-dir "$DATA_DIR" --index-dir "$INDEX_DIR" --top-k "$TOP_K"
        ;;

    api)
        $PY "$API_SCRIPT" --host "$API_HOST" --port "$API_PORT"
        ;;

    help|--help|"")
        usage
        ;;

    *)
        echo "[ERROR] Comando desconocido: $1"
        usage
        ;;
esac
