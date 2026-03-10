const PLUGIN_ID = "rag-search";
const TOOL_NAME = "rag_search";

type PluginConfig = {
  baseUrl: string;
  timeoutMs: number;
  topK: number;
};

const DEFAULT_CONFIG: PluginConfig = {
  baseUrl: "http://127.0.0.1:8000",
  timeoutMs: 30000,
  topK: 3
};

function clampTopK(value: unknown, fallback: number): number {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(1, Math.min(10, Math.trunc(n)));
}

function resolveConfig(api: any): PluginConfig {
  const entryConfig = api?.config?.plugins?.entries?.[PLUGIN_ID]?.config;
  const raw =
    entryConfig && typeof entryConfig === "object" && !Array.isArray(entryConfig)
      ? (entryConfig as Record<string, unknown>)
      : {};
  const baseUrlRaw = raw.baseUrl;
  const timeoutRaw = raw.timeoutMs;
  return {
    baseUrl: typeof baseUrlRaw === "string" && baseUrlRaw.trim()
      ? baseUrlRaw.trim()
      : DEFAULT_CONFIG.baseUrl,
    timeoutMs: Number.isFinite(Number(timeoutRaw))
      ? Math.max(1000, Math.min(120000, Number(timeoutRaw)))
      : DEFAULT_CONFIG.timeoutMs,
    topK: clampTopK(raw.topK, DEFAULT_CONFIG.topK)
  };
}

function formatResults(results: Array<{ content: string; source?: string; chunk_id?: number | string }>): string {
  if (!results.length) {
    return "No se encontraron fragmentos relevantes.";
  }

  const lines: string[] = [];
  results.forEach((item, idx) => {
    const source = item.source ?? "desconocido";
    const chunk = item.chunk_id ?? "n/a";
    lines.push(`[${idx + 1}] Fuente: ${source} | Chunk: ${chunk}`);
    lines.push(item.content);
    lines.push("---");
  });
  return lines.join("\n");
}

export default function register(api: any) {
  api.registerTool({
    name: TOOL_NAME,
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
        top_k: {
          type: "integer",
          minimum: 1,
          maximum: 10,
          description: "Cantidad de fragmentos a recuperar."
        },
        rebuild: {
          type: "boolean",
          description: "Si true, reconstruye el indice antes de buscar."
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
      const rebuild = Boolean(params?.rebuild);
      const endpoint = `${cfg.baseUrl.replace(/\/+$/, "")}/search`;

      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), cfg.timeoutMs);

      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query,
            top_k: topK,
            rebuild
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
}
