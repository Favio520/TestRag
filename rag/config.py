EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 180
TOP_K_DEFAULT = 4
BATCH_SIZE = 128
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".md", ".csv", ".xlsx"}
MANIFEST_FILENAME = "data_manifest.json"
SEARCH_TYPE_DEFAULT = "similarity_with_score"
MMR_FETCH_K_MULTIPLIER = 4
MMR_LAMBDA_MULT = 0.7
INGEST_MODE_DEFAULT = "incremental"

