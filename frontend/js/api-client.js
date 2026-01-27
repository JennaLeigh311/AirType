/**
 * AirType - API Client Module
 * Handles all REST API communication with the backend
 */

import Config from './config.js';

class ApiClient {
    constructor() {
        this.baseUrl = Config.API_BASE_URL;
        this.token = localStorage.getItem(Config.STORAGE.AUTH_TOKEN);
    }
    
    /**
     * Get authorization headers
     */
    getAuthHeaders() {
        const headers = {
            'Content-Type': 'application/json'
        };
        
        if (this.token) {
            headers['Authorization'] = `Bearer ${this.token}`;
        }
        
        return headers;
    }
    
    /**
     * Set authentication token
     */
    setToken(token) {
        this.token = token;
        if (token) {
            localStorage.setItem(Config.STORAGE.AUTH_TOKEN, token);
        } else {
            localStorage.removeItem(Config.STORAGE.AUTH_TOKEN);
        }
    }
    
    /**
     * Clear authentication
     */
    clearAuth() {
        this.token = null;
        localStorage.removeItem(Config.STORAGE.AUTH_TOKEN);
        localStorage.removeItem(Config.STORAGE.USER_DATA);
    }
    
    /**
     * Make API request
     */
    async request(endpoint, options = {}) {
        const url = `${this.baseUrl}${endpoint}`;
        
        const config = {
            headers: this.getAuthHeaders(),
            ...options
        };
        
        try {
            const response = await fetch(url, config);
            
            // Handle 401 Unauthorized
            if (response.status === 401) {
                this.clearAuth();
                window.dispatchEvent(new CustomEvent('auth:unauthorized'));
            }
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new ApiError(data.message || 'Request failed', response.status, data);
            }
            
            return data;
        } catch (error) {
            if (error instanceof ApiError) {
                throw error;
            }
            throw new ApiError(error.message || 'Network error', 0);
        }
    }
    
    /**
     * GET request
     */
    async get(endpoint, params = {}) {
        const queryString = new URLSearchParams(params).toString();
        const url = queryString ? `${endpoint}?${queryString}` : endpoint;
        return this.request(url, { method: 'GET' });
    }
    
    /**
     * POST request
     */
    async post(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'POST',
            body: JSON.stringify(data)
        });
    }
    
    /**
     * PUT request
     */
    async put(endpoint, data = {}) {
        return this.request(endpoint, {
            method: 'PUT',
            body: JSON.stringify(data)
        });
    }
    
    /**
     * DELETE request
     */
    async delete(endpoint) {
        return this.request(endpoint, { method: 'DELETE' });
    }
    
    // ==================
    // Auth Endpoints
    // ==================
    
    /**
     * Register new user
     */
    async register(username, email, password) {
        const response = await this.post('/users/register', {
            username,
            email,
            password
        });
        
        if (response.access_token) {
            this.setToken(response.access_token);
        }
        
        return response;
    }
    
    /**
     * Login user
     */
    async login(email, password) {
        const response = await this.post('/users/login', {
            email,
            password
        });
        
        if (response.access_token) {
            this.setToken(response.access_token);
            localStorage.setItem(Config.STORAGE.USER_DATA, JSON.stringify(response.user));
        }
        
        return response;
    }
    
    /**
     * Logout user
     */
    async logout() {
        try {
            await this.post('/users/logout');
        } finally {
            this.clearAuth();
        }
    }
    
    /**
     * Get current user profile
     */
    async getProfile() {
        return this.get('/users/me');
    }
    
    /**
     * Update user profile
     */
    async updateProfile(data) {
        return this.put('/users/me', data);
    }
    
    /**
     * Check if authenticated
     */
    isAuthenticated() {
        return !!this.token;
    }
    
    // ==================
    // Stroke Endpoints
    // ==================
    
    /**
     * Create stroke session
     */
    async createStrokeSession(deviceInfo = {}) {
        return this.post('/strokes/sessions', { device_info: deviceInfo });
    }
    
    /**
     * Get stroke session
     */
    async getStrokeSession(sessionId) {
        return this.get(`/strokes/sessions/${sessionId}`);
    }
    
    /**
     * Complete stroke session
     */
    async completeStrokeSession(sessionId) {
        return this.post(`/strokes/sessions/${sessionId}/complete`);
    }
    
    /**
     * Add stroke points
     */
    async addStrokePoints(sessionId, points) {
        return this.post(`/strokes/sessions/${sessionId}/points`, { points });
    }
    
    /**
     * Get user stroke sessions
     */
    async getUserSessions(params = {}) {
        return this.get('/strokes/sessions', params);
    }
    
    // ==================
    // Prediction Endpoints
    // ==================

    /**
     * Predict character from features (public endpoint)
     */
    async predict(features, topK = 5) {
        const url = `${this.baseUrl}/predictions/predict`;
        
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ features, top_k: topK })
            });
            
            const data = await response.json();
            
            if (!response.ok) {
                throw new ApiError(data.error || 'Prediction failed', response.status, data);
            }
            
            return data;
        } catch (error) {
            if (error instanceof ApiError) {
                throw error;
            }
            throw new ApiError(error.message || 'Network error', 0);
        }
    }

    /**
     * Get predictions for session
     */
    async getSessionPredictions(sessionId) {
        return this.get(`/predictions/sessions/${sessionId}`);
    }
    
    /**
     * Get user prediction history
     */
    async getPredictionHistory(params = {}) {
        return this.get('/predictions/history', params);
    }
    
    /**
     * Get prediction statistics
     */
    async getPredictionStats(period = 'week') {
        return this.get('/predictions/stats', { period });
    }
    
    // ==================
    // Health Endpoints
    // ==================
    
    /**
     * Check API health
     */
    async checkHealth() {
        // Health endpoint is at root, not under /api
        const url = this.baseUrl.replace('/api', '') + '/health';
        try {
            const response = await fetch(url);
            return await response.json();
        } catch (error) {
            throw new ApiError(error.message || 'Health check failed', 0);
        }
    }
    
    /**
     * Get detailed health status
     */
    async getDetailedHealth() {
        const url = this.baseUrl.replace('/api', '') + '/health/detailed';
        try {
            const response = await fetch(url);
            return await response.json();
        } catch (error) {
            throw new ApiError(error.message || 'Health check failed', 0);
        }
    }
}

/**
 * Custom API Error class
 */
class ApiError extends Error {
    constructor(message, status, data = null) {
        super(message);
        this.name = 'ApiError';
        this.status = status;
        this.data = data;
    }
}

// Export singleton instance
const apiClient = new ApiClient();
export default apiClient;
export { ApiError };
