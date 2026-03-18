# OpenClaw RAG Search Plugin

Plugin local para OpenClaw que expone la herramienta `rag_search` y consulta la API HTTP del proyecto RAG.

## 1) Levantar API RAG

En la raiz de este repositorio:

```bat
.\run.bat install
.\run.bat api
```

API esperada: `http://127.0.0.1:8000`

## 2) Instalar plugin en OpenClaw

```bat
openclaw plugins install -l D:\RAG\openclaw-rag-search-plugin

```

Si ya estaba instalado y quieres refrescar metadatos:

```bat
openclaw plugins uninstall rag-search
openclaw plugins install -l D:\RAG\openclaw-rag-search-plugin
```

## 3) Configurar plugin

### Windows / PowerShell

```powershell
openclaw config set plugins.entries.rag-search.enabled true --strict-json
openclaw config set plugins.entries.rag-search.config.baseUrl http://127.0.0.1:8000
openclaw config set plugins.entries.rag-search.config.topK 3 --strict-json
openclaw config set plugins.entries.rag-search.config.timeoutMs 30000 --strict-json
openclaw config set plugins.entries.rag-search.config.searchType mmr
openclaw config set plugins.entries.rag-search.config.chunkSize 900 --strict-json
openclaw config set plugins.entries.rag-search.config.chunkOverlap 180 --strict-json
openclaw config set plugins.allow[0] rag-search

```

Nota PowerShell: usa comillas simples `'...'` para JSON.

### Ubuntu / bash

En bash, los corchetes `[]` pueden ser interpretados por el shell. Pon la ruta completa entre comillas simples:

```bash
openclaw config set 'plugins.entries.rag-search.enabled' true --strict-json
openclaw config set 'plugins.entries.rag-search.config.baseUrl' http://127.0.0.1:8000
openclaw config set 'plugins.entries.rag-search.config.topK' 3 --strict-json
openclaw config set 'plugins.entries.rag-search.config.timeoutMs' 30000 --strict-json
openclaw config set 'plugins.entries.rag-search.config.searchType' mmr
openclaw config set 'plugins.entries.rag-search.config.chunkSize' 900 --strict-json
openclaw config set 'plugins.entries.rag-search.config.chunkOverlap' 180 --strict-json
openclaw config set 'plugins.allow[0]' rag-search
```

## 4) Habilitar herramienta en tu agente

Ejemplo para el primer agente:

### Windows / PowerShell

```powershell
openclaw config set agents.list[0].tools.allow '["rag-search"]' --strict-json
```

### Ubuntu / bash

```bash
openclaw config set 'agents.list[0].tools.allow' '["rag-search"]' --strict-json
```

Alternativa agregando entradas individuales:

```bash
openclaw config set 'agents.list[0].tools.allow[0]' group:core
openclaw config set 'agents.list[0].tools.allow[1]' rag-search
```

openclaw config set agents.list[0].tools.allow[0] group:core
openclaw config set agents.list[0].tools.allow[1] rag-search

Si quieres conservar herramientas core ademas del plugin:

```powershell
openclaw config set agents.list[0].tools.allow '["group:core","rag-search"]' --strict-json
```

```bash
openclaw config set 'agents.list[0].tools.allow' '["group:core","rag-search"]' --strict-json
```

## 5) Reiniciar gateway

```bat
openclaw gateway restart
```

## 6) Probar en chat

Pregunta al agente algo de tus manuales, por ejemplo:

`como se calibra el sensor de presion`

El agente podra llamar la tool `rag_search` y responder con los fragmentos recuperados.

## 7) Agregar mas archivos al RAG

Coloca tus documentos dentro de la carpeta `data/` del proyecto principal. Se admiten archivos `.pdf`, `.txt`, `.docx` y `.md`.

El sistema ahora detecta automaticamente si cambiaste, agregaste o eliminaste archivos en `data/` y reconstruye el indice FAISS en la siguiente consulta. No necesitas reiniciar la API para eso.

Si quieres forzar una reconstruccion manual de todos modos, puedes seguir usando `rebuild=true` desde la tool o el comando `run.bat rebuild`.

## 8) Estrategia de recuperacion

El plugin ahora puede elegir la estrategia de recuperacion del RAG:

- `mmr`: recomendada por defecto, reduce fragmentos repetidos.
- `similarity`: busqueda por similitud clasica.
- `similarity_with_score`: igual que similitud, pero el backend conserva el score en metadatos.

Ejemplo:

```bat
openclaw config set plugins.entries.rag-search.config.searchType mmr
```

## 9) Chunking para manuales tecnicos

El sistema ahora usa por defecto:

- `chunkSize = 900`
- `chunkOverlap = 180`

Estos valores suelen funcionar mejor en manuales tecnicos que un chunk muy pequeno, porque preservan mas contexto por fragmento.

Ejemplo:

```bat
openclaw config set plugins.entries.rag-search.config.chunkSize 900 --strict-json
openclaw config set plugins.entries.rag-search.config.chunkOverlap 180 --strict-json
```
