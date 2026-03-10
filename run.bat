@echo off
setlocal

cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"
set "PIP=.venv\Scripts\pip.exe"
set "SCRIPT=rag_system.py"
set "API_SCRIPT=rag_api.py"
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
if /I "%~1"=="ask" goto :ask
if /I "%~1"=="rebuild" goto :rebuild
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

:ask
shift
if "%~1"=="" (
    echo [ERROR] Debes enviar una consulta.
    echo Ejemplo: run.bat ask "como se calibra el sensor"
    exit /b 1
)
set "QUERY=%*"
"%PY%" "%SCRIPT%" "%QUERY%" --data-dir "%DATA_DIR%" --index-dir "%INDEX_DIR%" --top-k %TOP_K%
exit /b %ERRORLEVEL%

:rebuild
shift
if "%~1"=="" (
    echo [ERROR] Debes enviar una consulta.
    echo Ejemplo: run.bat rebuild "procedimiento de arranque"
    exit /b 1
)
set "QUERY=%*"
"%PY%" "%SCRIPT%" "%QUERY%" --data-dir "%DATA_DIR%" --index-dir "%INDEX_DIR%" --top-k %TOP_K% --rebuild
exit /b %ERRORLEVEL%

:api
"%PY%" "%API_SCRIPT%" --host "%API_HOST%" --port %API_PORT%
exit /b %ERRORLEVEL%

:usage
echo Uso:
echo   run.bat install
echo   run.bat ask "tu consulta"
echo   run.bat rebuild "tu consulta"
echo   run.bat api
echo.
echo Comandos:
echo   install  Instala dependencias desde requirements.txt
echo   ask      Consulta con el indice FAISS (si no existe, se crea)
echo   rebuild  Fuerza reconstruccion del indice y luego consulta
echo   api      Levanta la API local del RAG para integraciones (OpenClaw)
exit /b 1
