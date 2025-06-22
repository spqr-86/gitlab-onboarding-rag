# 📚 GitLab Onboarding Assistant

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-000000?style=for-the-badge&logo=database&logoColor=white)](https://www.trychroma.com/)

## 📋 Описание проекта

RAG-система для быстрого онбординга новых сотрудников GitLab. Вместо чтения 200+ страниц документации, сотрудники получают мгновенные ответы на свои вопросы через AI-ассистента.

### 🎯 Решаемая проблема

* **Было**: Необходимость тратить часы на поиск информации в объемной документации GitLab.
* **Стало**: Возможность получить точный ответ с указанием источника за несколько секунд.
* **Результат**: Ускорение процесса онбординга и повышение эффективности новых сотрудников.

### 📊 Ключевые метрики

- **Точность ответов**: 89% (тестировано на 50 реальных вопросах)
- **Скорость ответа**: 2.3 сек (оптимизировано с 8 сек)
- **Объем данных**: 847 страниц → 3,200 векторных чанков
- **Языковая поддержка**: Вопросы на RU/EN с единой базой знаний

## 🚀 Демо

![Demo Screenshot](docs/demo_screenshot.png)

### Примеры вопросов:

```
✅ Система отвечает на:
• "What are the six core values of GitLab?"
• "Как оформить отпуск?"
• "Describe the process for taking time off"
• "Какая философия у GitLab насчет boring solutions?"
• "How does the remote work policy work?"
• "Где найти информацию о бенефитах?"
```

## 🏗️ Архитектура

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Streamlit UI  │────▶│   rag_core.py    │────▶│   Gemini API    │
│   (Frontend)    │     │   (RAG Logic)    │     │   (Embeddings)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                          │
                               ▼                          ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │    ChromaDB      │     │   Gemini Flash  │
                        │  Vector Store    │     │   (Generation)  │
                        └──────────────────┘     └─────────────────┘
```

## 💻 Установка и запуск

### Требования
- Python 3.11+
- Docker (рекомендуется)
- Google AI Studio API Key

### Запуск через Docker (рекомендуемый способ)

1.  **Создайте файл `.env`** в корне проекта и добавьте в него ваш ключ:
    ```
    GOOGLE_API_KEY=your-api-key-here
    ```

2.  **Соберите Docker-образ:**
    ```bash
    docker build -t gitlab-onboarding-rag .
    ```

3.  **Запустите контейнер:**
    ```bash
    docker run --rm -p 8501:8501 --env-file .env gitlab-onboarding-rag
    ```

4.  Откройте в браузере `http://localhost:8501`.

### Локальный запуск (для разработки)

1.  **Создайте и активируйте виртуальное окружение:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Установите переменную окружения с вашим ключом:**
    ```bash
    export GOOGLE_API_KEY="your-api-key-here"
    ```

4.  **Запустите приложение:**
    ```bash
    streamlit run app.py
    ```

### Запуск приложения

```bash
streamlit run app.py
```

Приложение откроется в браузере: http://localhost:8501

## 🔧 Конфигурация

### Настройка параметров в `app.py`:

```python
# Параметры чанкинга
CHUNK_SIZE = 1000        # Размер чанка
CHUNK_OVERLAP = 100      # Перекрытие чанков

# Параметры поиска
N_RESULTS = 3            # Количество чанков для контекста

# Модель
MODEL_NAME = 'gemini-1.5-flash'
EMBEDDING_MODEL = 'models/text-embedding-004'
```
### Логирование:
```python
# Включение детального логирования
import logging
logging.basicConfig(level=logging.INFO)
```

## 🧪 Тестирование
Проект покрыт автоматическими тестами с использованием `pytest`.

* `tests/test_accuracy.py`: Проверяет качество ответов по ключевым словам.
* `tests/test_performance.py`: Измеряет скорость генерации ответов.

Для запуска всех тестов выполните команду в корне проекта (убедитесь, что venv активировано и ключ API установлен):
```bash
pytest -v
```

## 🚢 Roadmap

- [ ] Поддержка других форматов (DOCX, TXT, MD)
- [ ] Мультиязычный UI (полностью на русском)
- [ ] Расширить тестовый набор QA_DATASET для более точной оценки качества
- [ ] Улучшить логику извлечения источников, чтобы указывать не только файл, но и номер страницы
- [ ] Добавить в UI возможность выбора языка ответа (русский/английский)
- [ ] Реализовать историю чата для сохранения контекста беседы
- [ ] Экспорт ответов в PDF
- [ ] Fine-tuning на корпоративном стиле
- [ ] Интеграция с Slack/Teams
- [ ] A/B тестирование ответов

### Приоритетные улучшения:
- Улучшение качества чанкинга
- Оптимизация промптов
- Добавление новых источников данных
- UI/UX улучшения

## 📚 Использованные технологии

- **Streamlit** - быстрое создание веб-интерфейса
- **ChromaDB** - векторная база данных
- **Google Gemini** - эмбеддинги и генерация
- **PyPDF** - обработка PDF документов

