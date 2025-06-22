import os
import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pypdf

from config import (
    PDF_FOLDER_PATH, DB_PATH, COLLECTION_NAME, CHUNK_SIZE, 
    CHUNK_OVERLAP, N_RESULTS, EMBEDDING_MODEL
)
from prompts import PROMPT_TEMPLATE


load_dotenv()

# Класс для интеграции эмбеддингов Gemini с ChromaDB
class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Класс для интеграции эмбеддингов Gemini с ChromaDB.
    """
    def __init__(self, model_name='models/text-embedding-004', task_type="retrieval_document"):
        self.model_name = model_name
        self.task_type = task_type

    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/text-embedding-004'
        title = "Custom query"
        return genai.embed_content(model=EMBEDDING_MODEL,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]

# --- ФУНКЦИИ ДЛЯ RAG ---
# Функция для загрузки и обработки PDF
def load_and_process_pdfs_with_metadata(folder_path: str):
    st.info(f"Загрузка всех PDF из папки {folder_path}...")
    
    all_chunks = []
    all_metadatas = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    for filename in os.listdir():
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            st.write(f"Обработка файла: {filename}")
            try:
                reader = pypdf.PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                
                # Нарезаем на чанки текст ОДНОГО документа
                chunks = text_splitter.split_text(text)
                
                # Для каждого чанка создаем метаданные с именем файла
                for chunk in chunks:
                    all_chunks.append(chunk)
                    # В метаданных может быть любая полезная информация
                    all_metadatas.append({'source': filename})

            except Exception as e:
                st.error(f"Не удалось прочитать файл {filename}: {e}")

    st.info(f"Все документы разделены на {len(all_chunks)} частей (чанков).")
    return all_chunks, all_metadatas

# Функция для настройки и заполнения векторной базы данных
# @st.cache_resource - этот декоратор кэширует результат функции.
# Это значит, что PDF будет обрабатываться и база создаваться только один раз при первом запуске.
@st.cache_resource
def setup_database():
    st.info("Инициализация базы данных...")
    client = chromadb.PersistentClient(path=DB_PATH)
    embedding_function = GeminiEmbeddingFunction()
    
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    
    # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ ---
    # Проверяем, есть ли уже документы в коллекции
    if collection.count() > 0:
        st.sidebar.success("База данных успешно загружена из локального хранилища.")
        return collection
    
    # Если коллекция пуста, то выполняем полную загрузку
    st.sidebar.warning("База данных не найдена. Запускаю полную индексацию документов. Это может занять несколько минут...")
    
    chunks, metadatas = load_and_process_pdfs_with_metadata(PDF_FOLDER_PATH)
    
    st.info("Добавление чанков в базу данных...")
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        metadatas=metadatas
    )
    st.sidebar.success("База данных успешно создана и готова к работе!")
    return collection


def get_response(user_query: str, collection, model) -> tuple[str, str]:
    """
    Выполняет полный RAG-цикл:
    1. Находит релевантные чанки в базе данных.
    2. Создает промпт для LLM, включая найденные чанки.
    3. Генерирует ответ с помощью LLM.
    4. Возвращает сгенерированный ответ и источники.
    """
    
    # 1. Извлечение (Retrieval)
    results = collection.query(
        query_texts=[user_query],
        n_results=N_RESULTS,
        include=['documents', 'metadatas']
    )
    retrieved_docs = results['documents'][0]
    
    # Готовим источники для отображения пользователю
    sources_text = ""
    retrieved_metadatas = results['metadatas'][0]
    for i, doc in enumerate(retrieved_docs):
        source = retrieved_metadatas[i]['source']
        sources_text += f"**Источник: {source}**\n"
        sources_text += f"{doc}\n\n---\n\n"

    # 2. Дополнение (Augmentation) - Создание промпта
    prompt = PROMPT_TEMPLATE.format(
        retrieved_docs=retrieved_docs,
        user_query=user_query
    )

    # 3. Генерация (Generation)
    try:
        final_response = model.generate_content(prompt)
        return final_response.text, sources_text
    except Exception as e:
        return f"Произошла ошибка при генерации ответа: {e}", ""
