# --- Настройки путей ---
PDF_FOLDER_PATH = "data/"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "gitlab_handbook_collection"

# --- Параметры RAG ---
# Параметры для разделения текста на чанки
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# Количество извлекаемых чанков для контекста
N_RESULTS = 3

# --- Параметры моделей ---
EMBEDDING_MODEL = 'models/text-embedding-004'
GENERATIVE_MODEL = 'gemini-1.5-flash'