import streamlit as st
import google.generativeai as genai
import time

# --- КОНФИГУРАЦИЯ МОДЕЛИ ---
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model = genai.GenerativeModel('gemini-1.5-flash')
    model_initialized = True
except Exception as e:
    st.error(f"Ошибка инициализации модели: {e}")
    model_initialized = False


# --- ЛОГИКА ПРИЛОЖЕНИЯ ---
def get_response(user_query: str) -> str:
    """
    Принимает вопрос пользователя и возвращает ответ от модели Gemini.
    """
    try:
        response = model.generate_content(user_query)
        return response.text
    except Exception as e:
        return f"Произошла ошибка при обращении к AI: {e}"


# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ (UI) ---
st.title("GitLab Onboarding Assistant 🚀")
st.info("Этот бот поможет вам найти информацию в корпоративной документации GitLab.")

query = st.text_input("Задайте ваш вопрос:", placeholder="Например: как создать новый merge request?")

# Кнопка будет активна только если модель успешно инициализировалась
if st.button("Отправить", type="primary", disabled=not model_initialized):
    if query:
        with st.spinner("Думаю..."):
            response = get_response(query)
        st.success("Ответ:")
        st.markdown(response)
    else:
        st.warning("Пожалуйста, введите ваш вопрос.")
        
# Проверка инициализации модели
if not model_initialized:
    st.error("Не удалось загрузить модель. Проверьте ваш API ключ в 'secrets.toml'")
