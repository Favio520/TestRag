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

```bat
openclaw config set plugins.entries.rag-search.enabled true --strict-json
openclaw config set plugins.entries.rag-search.config.baseUrl http://127.0.0.1:8000
openclaw config set plugins.entries.rag-search.config.topK 3 --strict-json
openclaw config set plugins.entries.rag-search.config.timeoutMs 30000 --strict-json
openclaw config set plugins.allow[0] rag-search

```

Nota PowerShell: usa comillas simples `'...'` para JSON.

## 4) Habilitar herramienta en tu agente

Ejemplo para el primer agente:

```bat
openclaw config set agents.list[0].tools.allow "[\"rag-search\"]" --strict-json
```

openclaw config set agents.list[0].tools.allow[0] group:core
openclaw config set agents.list[0].tools.allow[1] rag-search

Si quieres conservar herramientas core ademas del plugin:

```bat
openclaw config set agents.list[0].tools.allow "[\"group:core\",\"rag-search\"]" --strict-json
```

## 5) Reiniciar gateway

```bat
openclaw gateway restart
```

## 6) Probar en chat

Pregunta al agente algo de tus manuales, por ejemplo:

`como se calibra el sensor de presion`

El agente podra llamar la tool `rag_search` y responder con los fragmentos recuperados.
