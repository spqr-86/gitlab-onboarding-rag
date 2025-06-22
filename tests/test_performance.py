import pytest
import os
import sys
import time
import statistics

# --- Стандартный код для импорта ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rag_core import get_response, setup_database
import google.generativeai as genai

# --- Вопросы для тестирования производительности ---
# Берем те же вопросы, что и для теста точности.
PERFORMANCE_QUESTIONS = [
    "What are the six core values of GitLab?",
    "How should you give negative feedback?",
    "How long is the initial onboarding period at GitLab?",
    "What is GitLab's philosophy on 'boring solutions'?",
    "What is the process for taking time off for five or more consecutive days?"
]

# --- Фикстуры для подготовки ресурсов (модель и БД) ---
# Они такие же, как в test_accuracy.py, pytest их переиспользует.
@pytest.fixture(scope="session")
def generative_model():
    print("\nConfiguring generative model for performance test...")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        pytest.fail("GOOGLE_API_KEY environment variable not set.")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

@pytest.fixture(scope="session")
def db_collection(generative_model):
    print("\nSetting up database for performance test...")
    collection = setup_database("data/")
    return collection

# --- Основная логика теста производительности ---
def test_response_performance(db_collection, generative_model):
    """
    Этот тест измеряет время ответа для нескольких вопросов и выводит статистику.
    """
    response_times = []

    print("\n--- Running Performance Test ---")
    for i, question in enumerate(PERFORMANCE_QUESTIONS):
        start_time = time.monotonic()
        
        get_response(question, db_collection, generative_model) # Нам не важен результат, только время
        
        end_time = time.monotonic()
        duration = end_time - start_time
        response_times.append(duration)
        print(f"Question #{i+1}: '{question[:30]}...' - Time: {duration:.2f}s")

    # Рассчитываем и выводим статистику
    min_time = min(response_times)
    max_time = max(response_times)
    avg_time = statistics.mean(response_times)
    
    print("\n--- Performance Summary ---")
    print(f"Total questions tested: {len(PERFORMANCE_QUESTIONS)}")
    print(f"Average response time: {avg_time:.2f}s")
    print(f"Fastest response time: {min_time:.2f}s")
    print(f"Slowest response time: {max_time:.2f}s")
    print("---------------------------")

    # Тест считается пройденным, если среднее время меньше определенного порога (например, 10 секунд)
    assert avg_time < 10.0, f"Average response time {avg_time:.2f}s is over the 10s threshold."