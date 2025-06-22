import pytest
import os
import sys
import google.generativeai as genai

# --- Код для импорта, как и раньше ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_core import get_response, setup_database

# --- Тестовый набор данных (остается без изменений) ---
QA_DATASET = [
    {
        "question": "What are the six core values of GitLab?",
        # Эти слова, скорее всего, останутся на английском, так как это названия
        "expected_keywords": ["Collaboration", "Results", "Efficiency", "Diversity", "Iteration", "Transparency"],
        "source_file": "GitLab Values _ The GitLab Handbook.pdf"
    },
    {
        "question": "How should you give negative feedback?",
        "expected_keywords": ["один", "наедине", "минимальной", "маленькой", "1-1"],
        "source_file": "GitLab Values _ The GitLab Handbook.pdf"
    },  
    {
        "question": "How long is the initial onboarding period at GitLab?",
        "expected_keywords": ["две полные недели", "2 недели", "третьей неделе"],
        "source_file": "GitLab Onboarding _ The GitLab Handbook.pdf"
    },
    {
        "question": "What is GitLab's philosophy on 'boring solutions'?",
        "expected_keywords": ["простым", "скучных решений", "сложностью"],
        "source_file": "GitLab Values _ The GitLab Handbook.pdf"
    }
]

# --- ИСПРАВЛЕНИЕ: ДОБАВЛЯЕМ ФИКСТУРУ ДЛЯ МОДЕЛИ ---

# Эта фикстура будет создавать модель ОДИН раз за сессию тестов
@pytest.fixture(scope="session")
def generative_model():
    print("\nConfiguring generative model...")
    # Здесь нужно безопасно получить ключ API.
    # Pytest автоматически подхватывает переменные окружения.
    # Перед запуском теста установите переменную: export GOOGLE_API_KEY="ваш_ключ"
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.fail("GOOGLE_API_KEY environment variable not set.")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# Фикстура для БД, как и раньше, но теперь зависит от модели
@pytest.fixture(scope="session")
def db_collection(generative_model): # Передаем фикстуру модели, чтобы genai был настроен
    print("\nSetting up database for testing...")
    collection = setup_database("data/")
    return collection

# --- ИСПРАВЛЕНИЕ: ОБНОВЛЯЕМ ТЕСТ, ЧТОБЫ ОН ПРИНИМАЛ МОДЕЛЬ ---

@pytest.mark.parametrize("qa_pair", QA_DATASET)
def test_Youtubeing_and_sourcing(qa_pair, db_collection, generative_model): # Добавили generative_model
    """
    Этот тест проверяет две вещи:
    1. Ответ содержит ожидаемые ключевые слова.
    2. Источник, найденный моделью, соответствует файлу, где находится ответ.
    """
    question = qa_pair["question"]
    expected_keywords = qa_pair["expected_keywords"]
    expected_source = qa_pair["source_file"]

    # ИСПРАВЛЕНО: Передаем модель в функцию
    response_text, sources_text = get_response(question, db_collection, generative_model)

    # Проверки остаются такими же
    found_any_keyword = any(keyword.lower() in response_text.lower() for keyword in expected_keywords)
    assert found_any_keyword, (
        f"FAIL: For question '{question}', no expected keywords {expected_keywords} found in response.\n"
        f"--> Response was: '{response_text}'"
    )

    assert expected_source in sources_text, (
        f"FAIL: For question '{question}', expected source '{expected_source}' was not found in sources.\n"
        f"--> Sources were: '{sources_text}'"
    )