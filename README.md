# OpenClaw RAG Search Plugin

## RAG search focused on simple documents without advanced metadata
## Fast search with an approximate cost of 45k tokens through OpenClaw
## Supported document types: `csv`, `docx`, `md`, `pdf`, `txt`, `xlsx`
## Supports up to 5 users asking at the same time
## Stored FAISS data cannot be manually manipulated
## Indexed data cannot be selectively removed from FAISS; it is removed only when reindexing

Local OpenClaw plugin that exposes the `rag_search` tool and queries the RAG project's HTTP API.

# Installation

## 1) Start the RAG API

From the root of this repository:

Windows

```bat
.\run.bat install
.\run.bat api
```

Linux / macOS

```sh
./run.sh install
./run.sh api
```

Expected API URL: `http://127.0.0.1:8000`

## 2) Install the plugin in OpenClaw

```bat
openclaw plugins install -l D:\RAG\openclaw-rag-search-plugin
```

If it is already installed and you want to refresh its metadata:

```bat
openclaw plugins uninstall rag-search
openclaw plugins install -l D:\RAG\openclaw-rag-search-plugin
```

## 3) Configure the plugin

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

PowerShell note: use single quotes `'...'` for JSON values.

### Ubuntu / bash

In bash, square brackets `[]` may be interpreted by the shell. Put the full path in single quotes:

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

## 4) Enable the tool in your agent

Example for the first agent:

### Windows / PowerShell

```powershell
openclaw config set agents.list[0].tools.allow '["rag-search"]' --strict-json
```

### Ubuntu / bash

```bash
openclaw config set 'agents.list[0].tools.allow' '["rag-search"]' --strict-json
```

Alternative by adding entries individually:

```bash
openclaw config set 'agents.list[0].tools.allow[0]' group:core
openclaw config set 'agents.list[0].tools.allow[1]' rag-search
```

If you want to keep core tools in addition to the plugin:

```powershell
openclaw config set agents.list[0].tools.allow '["group:core","rag-search"]' --strict-json
```

```bash
openclaw config set 'agents.list[0].tools.allow' '["group:core","rag-search"]' --strict-json
```

## 5) Restart the gateway

```bat
openclaw gateway restart
```

## 6) Test it in chat

Ask the agent something from your manuals, for example:

`how do you calibrate the pressure sensor`

The agent will be able to call the `rag_search` tool and respond using the retrieved fragments.

## 7) Add more files to the RAG

Place your documents inside the `data/` folder in the main project. Supported file types are `.pdf`, `.txt`, `.docx`, `.md`, `.csv`, and `.xlsx`.

The current system can rebuild the FAISS index when new files are ingested. Depending on your current backend flow, you may need to run ingestion manually before searching.

## 8) Optional document filter

The search endpoint and the `rag_search` tool accept an optional `document` field. You can use it to restrict retrieval to a single file by name, relative path, or title.

Examples:

- `manual_bomba_hidraulica.md`
- `subfolder/manual_bomba_hidraulica.md`
- `manual_bomba_hidraulica`

## 9) Retrieval strategy

The plugin can choose the RAG retrieval strategy:

- `mmr`: recommended by default, reduces repeated fragments
- `similarity`: classic similarity search
- `similarity_with_score`: same as similarity search, but the backend also keeps the score in metadata
- `hybrid`: combines vector retrieval with a lexical layer, which is useful for exact names, codes, and technical terms

The backend also supports:

- `scoreThreshold`: if the best result is too weak, the RAG returns no evidence instead of forcing a bad answer
- `rerank`: applies a second ranking pass over the best candidates and usually improves final ordering

Example:

```bat
openclaw config set plugins.entries.rag-search.config.searchType mmr
openclaw config set plugins.entries.rag-search.config.scoreThreshold 0.35 --strict-json
openclaw config set plugins.entries.rag-search.config.rerank true --strict-json
```

## 10) Chunking for technical manuals

The system currently uses these defaults:

- `chunkSize = 900`
- `chunkOverlap = 180`

These values usually work better for technical manuals than very small chunks, because they preserve more context per fragment.

Example:

```bat
openclaw config set plugins.entries.rag-search.config.chunkSize 900 --strict-json
openclaw config set plugins.entries.rag-search.config.chunkOverlap 180 --strict-json
```
