@echo off
setlocal

cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "PIP=.venv\Scripts\pip.exe"
set "SCRIPT=rag_system.py"
set "API_SCRIPT=rag_api.py"
set "WATCH_SCRIPT=watch_rag.py"
set "DATA_DIR=data"
set "INDEX_DIR=faiss_index"
set "TOP_K=3"
set "API_HOST=127.0.0.1"
set "API_PORT=8000"

if not exist "%PY%" (
    echo [ERROR] No se encontro el entorno virtual en ".venv".
    echo Crea uno con: py -m venv .venv
    exit /b 1
)

if "%~1"=="" goto :usage

if /I "%~1"=="install" goto :install
if /I "%~1"=="ingest" goto :ingest
if /I "%~1"=="ingest-full" goto :ingestfull
if /I "%~1"=="watch" goto :watch
if /I "%~1"=="ask" goto :ask
if /I "%~1"=="api" goto :api
if /I "%~1"=="help" goto :usage
if /I "%~1"=="--help" goto :usage

goto :usage

:install
if not exist "requirements.txt" (
    echo [ERROR] No se encontro requirements.txt
    exit /b 1
)
"%PIP%" install -r requirements.txt
exit /b %ERRORLEVEL%

:ingest
"%PY%" "%SCRIPT%" --ingest --ingest-mode incremental --data-dir "%DATA_DIR%" --index-dir "%INDEX_DIR%" --chunk-size 900 --chunk-overlap 180
exit /b %ERRORLEVEL%

:ingestfull
"%PY%" "%SCRIPT%" --ingest --ingest-mode full --data-dir "%DATA_DIR%" --index-dir "%INDEX_DIR%" --chunk-size 900 --chunk-overlap 180
exit /b %ERRORLEVEL%

:watch
"%PY%" "%WATCH_SCRIPT%" --data-dir "%DATA_DIR%" --index-dir "%INDEX_DIR%" --debounce-seconds 10 --stability-seconds 2 --ingest-mode incremental
exit /b %ERRORLEVEL%

:ask
shift
if "%~1"=="" (
    echo [ERROR] Debes enviar una consulta.
    echo Ejemplo: run.bat ask "como se calibra el sensor" --document manual_bomba_hidraulica.md
    exit /b 1
)
set "QUERY="
set "DOCUMENT="
:ask_parse
if "%~1"=="" goto :ask_run
if /I "%~1"=="--document" (
    shift
    if "%~1"=="" (
        echo [ERROR] Debes indicar un documento despues de --document.
        exit /b 1
    )
    set "DOCUMENT=%~1"
    shift
    goto :ask_parse
)
if defined QUERY (
    set "QUERY=%QUERY% %~1"
) else (
    set "QUERY=%~1"
)
shift
goto :ask_parse

:ask_run
if not defined QUERY (
    echo [ERROR] Debes enviar una consulta.
    exit /b 1
)
if defined DOCUMENT (
    "%PY%" "%SCRIPT%" "%QUERY%" --data-dir "%DATA_DIR%" --index-dir "%INDEX_DIR%" --top-k %TOP_K% --document "%DOCUMENT%"
) else (
    "%PY%" "%SCRIPT%" "%QUERY%" --data-dir "%DATA_DIR%" --index-dir "%INDEX_DIR%" --top-k %TOP_K%
)
exit /b %ERRORLEVEL%

:api
"%PY%" "%API_SCRIPT%" --host "%API_HOST%" --port %API_PORT%
exit /b %ERRORLEVEL%

:usage
echo Uso:
echo   run.bat install
echo   run.bat ingest
echo   run.bat ingest-full
echo   run.bat watch
echo   run.bat ask "tu consulta" --document manual.md
echo   run.bat api
echo.
echo Comandos:
echo   install  Instala dependencias desde requirements.txt
echo   ingest   Actualiza el indice FAISS de forma incremental
echo   ingest-full  Reconstruye el indice FAISS completo
echo   watch    Observa data y dispara ingesta incremental automatica
echo   ask      Consulta con el indice FAISS ya construido
echo   api      Levanta la API local del RAG para integraciones (OpenClaw)
exit /b 1
