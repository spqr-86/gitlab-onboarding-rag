import os
import streamlit as st
import google.generativeai as genai
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pypdf
import time


# Класс для интеграции эмбеддингов Gemini с ChromaDB
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = 'models/text-embedding-004'
        title = "Custom query"
        return genai.embed_content(model=model,
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
        chunk_size=1000,
        chunk_overlap=100
    )

    for filename in os.listdir(folder_path):
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
def setup_database(folder_path):
    # Получаем и чанки, и метаданные
    chunks, metadatas = load_and_process_pdfs_with_metadata(folder_path)
    
    st.info("Создание и настройка векторной базы данных (ChromaDB)...")
    client = chromadb.Client()
    embedding_function = GeminiEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="gitlab_handbook_collection",
        embedding_function=embedding_function
    )
    
    st.info("Добавление чанков в базу данных...")
    # Используем специальный параметр 'metadatas' при добавлении
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        metadatas=metadatas # <-- Вот здесь мы передаем метаданные
    )
    st.success("База данных готова к работе!")
    return collection


# --- ОСНОВНАЯ ЛОГИКА ПРИЛОЖЕНИЯ ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    PDF_FOLDER_PATH = "data/"
    db_collection = setup_database(PDF_FOLDER_PATH)
    model_initialized = True
except Exception as e:
    st.error(f"Ошибка инициализации модели: {e}")
    model_initialized = False


def get_response(user_query: str, collection) -> tuple[str, str]:
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
        n_results=3, # Берем 3 наиболее релевантных чанка
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
    prompt_template = f"""
    Ты — полезный и информативный ассистент по онбордингу в компании GitLab. 
    Твоя задача — отвечать на вопросы новых сотрудников, основываясь ИСКЛЮЧИТЕЛЬНО на предоставленных тебе фрагментах текста на английском языке из внутренней документации ('PASSAGES').

    ***ВАЖНО: Твой финальный ответ всегда должен быть на РУССКОМ ЯЗЫКЕ.***

    Тон твоего ответа должен быть дружелюбным и профессиональным.
    Структурируй ответ так, чтобы его было легко читать. Используй списки, если это уместно.
    
    Если информация для ответа на вопрос отсутствует в предоставленных фрагментах, вежливо ответь по-русски: 
    "К сожалению, я не смог найти точную информацию по вашему вопросу в доступной мне документации."
    Не придумывай ничего от себя.

    Вот фрагменты документации на английском:
    PASSAGES:
    {retrieved_docs}

    А вот вопрос пользователя (он может быть на любом языке):
    QUESTION:
    {user_query}

    ANSWER (на русском языке):
    """

    # 3. Генерация (Generation)
    try:
        final_response = model.generate_content(prompt_template)
        return final_response.text, sources_text
    except Exception as e:
        return f"Произошла ошибка при генерации ответа: {e}", ""


# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ (UI) ---
st.title("GitLab Onboarding Assistant 🚀")
st.info("Бот, отвечающий на вопросы по документации GitLab.")

if model_initialized:
    st.sidebar.success(f"База данных успешно загружена. Количество документов: {db_collection.count()}")

query = st.text_input("Задайте ваш вопрос:", placeholder="Например: как создать новый merge request?")

if st.button("Отправить", type="primary", disabled=not model_initialized):
    if query:
        with st.spinner("Анализирую документы и генерирую ответ..."):
            # Теперь функция возвращает и ответ, и источники
            response, sources = get_response(query, db_collection)
        
        st.success("Ответ:")
        st.markdown(response)
        
        # Добавляем выпадающий список для просмотра источников
        with st.expander("Показать источники, на которых основан ответ"):
            st.markdown(sources)
    else:
        st.warning("Пожалуйста, введите ваш вопрос.")

if not model_initialized:
    st.error("Не удалось загрузить модель или базу данных. Проверьте API ключ и наличие PDF файла.")
