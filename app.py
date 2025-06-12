import streamlit as st

st.title("GitLab Onboarding Assistant 🚀")
st.info("Этот бот поможет вам найти информацию в корпоративной документации GitLab.")
query = st.text_input("Задайте ваш вопрос:", placeholder="Например: как создать новый merge request?")

if st.button("Отправить", type="primary"):
    if query:
        st.markdown(f"Вы спросили: **{query}**")
        with st.spinner("Думаю..."):
            st.success("Ответ будет здесь... (Это заглушка для Итерации 1)")
    else:
        st.warning("Пожалуйста, введите ваш вопрос.")
