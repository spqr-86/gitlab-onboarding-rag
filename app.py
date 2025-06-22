import streamlit as st
import google.generativeai as genai

from rag_core import setup_database, get_response


# --- ОСНОВНАЯ ЛОГИКА ПРИЛОЖЕНИЯ ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    PDF_FOLDER_PATH = "data/"
    db_collection = setup_database()
    model_initialized = True
except Exception as e:
    st.error(f"Ошибка инициализации модели: {e}")
    model_initialized = False


# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ (UI) ---
st.title("GitLab Onboarding Assistant 🚀")
st.info("Этот ассистент отвечает на вопросы по внутренней документации GitLab, используя RAG-технологию. Попробуйте спросить что-нибудь!")

# --- РЕШЕНИЕ ПРОБЛЕМЫ 1: Примеры вопросов ---
st.subheader("Или попробуйте один из этих примеров:")

# Создаем три колонки для кнопок
col1, col2, col3 = st.columns(3)

# Список вопросов
example_questions = [
    "What are the six core values of GitLab?",
    "Describe the process for taking time off.",
    "What is GitLab's philosophy on 'boring solutions'?"
]

# Функция для обработки нажатия кнопки
def run_query(question):
    with st.spinner("Анализирую документы и генерирую ответ..."):
        response, sources = get_response(question, db_collection, model)
    
    st.session_state.response = response
    st.session_state.sources = sources

with col1:
    if st.button(example_questions[0]):
        run_query(example_questions[0])
        
with col2:
    if st.button(example_questions[1]):
        run_query(example_questions[1])

with col3:
    if st.button(example_questions[2]):
        run_query(example_questions[2])

st.divider() # Горизонтальная черта для разделения

# --- Основное поле ввода ---
query = st.text_input("Задайте ваш вопрос здесь:", placeholder="Например: как происходит процесс онбординга?")

if st.button("Отправить", type="primary"):
    if query:
        run_query(query)
    else:
        st.warning("Пожалуйста, введите ваш вопрос.")

# Используем st.session_state, чтобы ответ не пропадал при взаимодействии с другими элементами
if 'response' in st.session_state:
    st.success("Ответ, сгенерированный на основе документов:")
    st.markdown(st.session_state.response)
    
    with st.expander("✅ Показать источники и проверить ответ"):
        st.markdown(st.session_state.sources)
