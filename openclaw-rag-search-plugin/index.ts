const PLUGIN_ID = "rag-search";
const SEARCH_TOOL_NAME = "rag_search";
const INGEST_TOOL_NAME = "rag_ingest";

type PluginConfig = {
  baseUrl: string;
  timeoutMs: number;
  topK: number;
  searchType: "mmr" | "similarity" | "similarity_with_score" | "hybrid";
  rerank: boolean;
  scoreThreshold: number;
  modelName: string;
  chunkSize: number;
  chunkOverlap: number;
};

const DEFAULT_CONFIG: PluginConfig = {
  baseUrl: "http://127.0.0.1:8000",
  timeoutMs: 30000,
  topK: 4,
  searchType: "mmr",
  rerank: false,
  scoreThreshold: 0.35,
  modelName: "sentence-transformers/all-MiniLM-L6-v2",
  chunkSize: 900,
  chunkOverlap: 180
};

function clampTopK(value: unknown, fallback: number): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(1, Math.min(10, Math.trunc(n)));
}

function clampChunkSize(value: unknown, fallback: number): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(100, Math.min(4000, Math.trunc(n)));
}

function clampChunkOverlap(value: unknown, fallback: number): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(1000, Math.trunc(n)));
}

function clampScoreThreshold(value: unknown, fallback: number): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(0, Math.min(1, n));
}

function resolveConfig(api: any): PluginConfig {
  const entryConfig = api?.config?.plugins?.entries?.[PLUGIN_ID]?.config;
  const raw =
    entryConfig && typeof entryConfig === "object" && !Array.isArray(entryConfig)
      ? (entryConfig as Record<string, unknown>)
      : {};
  const baseUrlRaw = raw.baseUrl;
  const timeoutRaw = raw.timeoutMs;
  const searchTypeRaw = raw.searchType;
  const modelNameRaw = raw.modelName;
  const rerankRaw = raw.rerank;
  const searchType =
    searchTypeRaw === "similarity" || searchTypeRaw === "similarity_with_score" || searchTypeRaw === "mmr" || searchTypeRaw === "hybrid"
      ? searchTypeRaw
      : DEFAULT_CONFIG.searchType;
  return {
    baseUrl: typeof baseUrlRaw === "string" && baseUrlRaw.trim()
      ? baseUrlRaw.trim()
      : DEFAULT_CONFIG.baseUrl,
    timeoutMs: Number.isFinite(Number(timeoutRaw))
      ? Math.max(1000, Math.min(120000, Number(timeoutRaw)))
      : DEFAULT_CONFIG.timeoutMs,
    topK: clampTopK(raw.topK, DEFAULT_CONFIG.topK),
    searchType,
    rerank: rerankRaw === true,
    scoreThreshold: clampScoreThreshold(raw.scoreThreshold, DEFAULT_CONFIG.scoreThreshold),
    modelName: typeof modelNameRaw === "string" && modelNameRaw.trim()
      ? modelNameRaw.trim()
      : DEFAULT_CONFIG.modelName,
    chunkSize: clampChunkSize(raw.chunkSize, DEFAULT_CONFIG.chunkSize),
    chunkOverlap: clampChunkOverlap(raw.chunkOverlap, DEFAULT_CONFIG.chunkOverlap)
  };
}

function formatResults(results: Array<{
  content: string;
  source?: string;
  source_name?: string;
  title?: string;
  doc_type?: string;
  section?: string;
  page_start?: number;
  page_end?: number;
  chunk_id?: number | string;
  score?: number;
  raw_score?: number;
  vector_score?: number;
  lexical_score?: number;
  rerank_score?: number;
  score_type?: string;
}>): string {
  if (!results.length) {
    return "No se encontraron fragmentos relevantes.";
  }

  const lines: string[] = [];
  results.forEach((item, idx) => {
    const source = item.source ?? "desconocido";
    const sourceName = item.source_name ?? source;
    const chunk = item.chunk_id ?? "n/a";
    const metaParts = [
      `Fuente: ${sourceName}`,
      item.title ? `Titulo: ${item.title}` : null,
      item.doc_type ? `Tipo: ${item.doc_type}` : null,
      item.section ? `Seccion: ${item.section}` : null,
      item.page_start
        ? item.page_end && item.page_end !== item.page_start
          ? `Paginas: ${item.page_start}-${item.page_end}`
          : `Pagina: ${item.page_start}`
        : null,
      typeof item.score === "number" ? `Score: ${item.score.toFixed(3)}` : null,
      typeof item.vector_score === "number" ? `Vector: ${item.vector_score.toFixed(3)}` : null,
      typeof item.lexical_score === "number" ? `Lexical: ${item.lexical_score.toFixed(3)}` : null,
      typeof item.rerank_score === "number" ? `Rerank: ${item.rerank_score.toFixed(3)}` : null,
      `Chunk: ${chunk}`
    ].filter(Boolean);
    lines.push(`[${idx + 1}] ${metaParts.join(" | ")}`);
    lines.push(item.content);
    lines.push("---");
  });
  return lines.join("\n");
}

export default function register(api: any) {
  api.registerTool({
    name: SEARCH_TOOL_NAME,
    label: "Buscar en manuales (RAG)",
    description: "Busca fragmentos relevantes en documentos locales indexados.",
    optional: true,
    parameters: {
      type: "object",
      additionalProperties: false,
      properties: {
        query: {
          type: "string",
          description: "Consulta del usuario."
        },
        document: {
          type: "string",
          description: "Filtro opcional por documento. Acepta nombre de archivo, ruta relativa o titulo."
        },
        top_k: {
          type: "integer",
          minimum: 1,
          maximum: 10,
          description: "Cantidad de fragmentos a recuperar."
        }
      },
      required: ["query"]
    },
    async execute(_callId: string, params: any) {
      const cfg = resolveConfig(api);
      const query = String(params?.query ?? "").trim();
      if (!query) {
        return {
          content: [{ type: "text", text: "Error: query vacia." }]
        };
      }

      const topK = clampTopK(params?.top_k, cfg.topK);
      const document = typeof params?.document === "string" ? params.document.trim() : "";
      const endpoint = `${cfg.baseUrl.replace(/\/+$/, "")}/search`;

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), cfg.timeoutMs);

      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            document: document || undefined,
            top_k: topK,
            search_type: cfg.searchType,
            rerank: cfg.rerank,
            score_threshold: cfg.scoreThreshold,
            model_name: cfg.modelName,
            chunk_size: cfg.chunkSize,
            chunk_overlap: cfg.chunkOverlap
          }),
          signal: controller.signal
        });

        if (!response.ok) {
          const body = await response.text();
          return {
            content: [{
              type: "text",
              text: `Error RAG (${response.status}): ${body}`
            }]
          };
        }

        const data = await response.json();
        const text = formatResults(Array.isArray(data?.results) ? data.results : []);
        return {
          content: [{ type: "text", text }]
        };
      } catch (error: any) {
        const reason = error?.name === "AbortError"
          ? "Timeout consultando la API RAG."
          : `No se pudo consultar la API RAG: ${String(error?.message ?? error)}`;
        return {
          content: [{ type: "text", text: reason }]
        };
      } finally {
        clearTimeout(timeout);
      }
    }
  });

  api.registerTool({
    name: INGEST_TOOL_NAME,
    label: "Actualizar indice RAG",
    description: "Ejecuta la ingesta del indice RAG. Usa incremental por defecto y full solo cuando quieras reconstruir todo.",
    optional: true,
    parameters: {
      type: "object",
      additionalProperties: false,
      properties: {
        mode: {
          type: "string",
          enum: ["incremental", "full"],
          description: "Modo de ingesta. Incremental agrega cambios nuevos; full reconstruye todo el indice."
        }
      }
    },
    async execute(_callId: string, params: any) {
      const cfg = resolveConfig(api);
      const mode = params?.mode === "full" ? "full" : "incremental";
      const endpoint = `${cfg.baseUrl.replace(/\/+$/, "")}/ingest`;

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), cfg.timeoutMs);

      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            mode,
            model_name: cfg.modelName,
            chunk_size: cfg.chunkSize,
            chunk_overlap: cfg.chunkOverlap
          }),
          signal: controller.signal
        });

        if (!response.ok) {
          const body = await response.text();
          return {
            content: [{
              type: "text",
              text: `Error de ingesta RAG (${response.status}): ${body}`
            }]
          };
        }

        const data = await response.json();
        const text = [
          "Ingesta completada.",
          `Modo: ${String(data?.mode ?? mode)}`,
          `Documentos detectados: ${String(data?.documents ?? "n/a")}`,
          `Archivos agregados: ${String(data?.added_files ?? 0)}`,
          `Archivos actualizados: ${String(data?.updated_files ?? 0)}`,
          `Archivos eliminados: ${String(data?.deleted_files ?? 0)}`
        ].join("\n");
        return {
          content: [{ type: "text", text }]
        };
      } catch (error: any) {
        const reason = error?.name === "AbortError"
          ? "Timeout ejecutando la ingesta RAG."
          : `No se pudo ejecutar la ingesta RAG: ${String(error?.message ?? error)}`;
        return {
          content: [{ type: "text", text: reason }]
        };
      } finally {
        clearTimeout(timeout);
      }
    }
  });
}
