import streamlit as st
import time


def get_response(user_query: str) -> str:
    """
    Принимает вопрос пользователя и возвращает ответ.
    В этой итерации используется "фейковая" логика.
    """
    # Имитируем небольшую задержку, как будто AI думает
    time.sleep(1) 
    return f"Ответ на вопрос '{user_query}' будет сгенерирован здесь. (Это заглушка для Итерации 2)"


# --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ (UI) ---
st.title("GitLab Onboarding Assistant 🚀")
st.info("Этот бот поможет вам найти информацию в корпоративной документации GitLab.")

query = st.text_input("Задайте ваш вопрос:", placeholder="Например: как создать новый merge request?")

if st.button("Отправить", type="primary"):
    if query:
        with st.spinner("Думаю..."):
            response = get_response(query)
        st.success(response)
    else:
        st.warning("Пожалуйста, введите ваш вопрос.")
