// Конфигурация API
const API_CONFIG = {
    BASE_URL: 'http://localhost:8000',
    ENDPOINTS: {
        CHAT: '/api/chat/message',
        HEALTH: '/api/health'
    }
};

// Утилиты для работы с сессиями
function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function getSessionId() {
    let sessionId = localStorage.getItem('titanic_session_id');
    if (!sessionId) {
        sessionId = generateSessionId();
        localStorage.setItem('titanic_session_id', sessionId);
    }
    return sessionId;
}

// Проверка здоровья API
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.HEALTH}`);
        const data = await response.json();
        console.log('API Status:', data);
        return data.status === 'healthy';
    } catch (error) {
        console.error('API недоступен:', error);
        return false;
    }
}