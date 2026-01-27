/**
 * AirType - Main Application Module
 * Orchestrates all components and handles UI interactions
 */

import Config from './config.js';
import apiClient from './api-client.js';
import videoStream from './video-stream.js';
import canvasRenderer from './canvas-renderer.js';
import wsClient from './websocket-client.js';

class App {
    constructor() {
        // DOM Elements
        this.elements = {};
        
        // State
        this.isInitialized = false;
        this.isVideoMode = false;
        this.predictionHistory = [];
        this.currentUser = null;
        
        // Debounce timer for predictions
        this.predictionDebounce = null;
    }
    
    /**
     * Initialize the application
     */
    async init() {
        try {
            // Get DOM elements
            this.cacheElements();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize modules
            await this.initializeModules();
            
            // Check authentication
            this.checkAuth();
            
            // Check API health
            this.checkApiHealth();
            
            this.isInitialized = true;
            console.log('AirType initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize AirType:', error);
            this.showToast('Failed to initialize application', 'error');
        }
    }
    
    /**
     * Cache DOM elements
     */
    cacheElements() {
        this.elements = {
            // Video section
            videoElement: document.getElementById('video-stream'),
            overlayCanvas: document.getElementById('overlay-canvas'),
            statusIndicator: document.getElementById('status-indicator'),
            statusText: document.getElementById('status-text'),
            startCameraBtn: document.getElementById('start-camera'),
            stopCameraBtn: document.getElementById('stop-camera'),
            
            // Drawing section
            drawingCanvas: document.getElementById('drawing-canvas'),
            clearCanvasBtn: document.getElementById('clear-canvas'),
            undoBtn: document.getElementById('undo-stroke'),
            predictBtn: document.getElementById('predict-btn'),
            
            // Results section
            predictionResult: document.getElementById('prediction-result'),
            noPrediction: document.getElementById('no-prediction'),
            predictionDisplay: document.getElementById('prediction-display'),
            predictedChar: document.getElementById('predicted-char'),
            confidenceFill: document.getElementById('confidence-fill'),
            confidencePercent: document.getElementById('confidence-percent'),
            alternativesList: document.getElementById('alternatives-list'),
            inferenceTime: document.getElementById('inference-time'),
            historyList: document.getElementById('history-list'),
            
            // Auth
            loginBtn: document.getElementById('login-btn'),
            userMenu: document.getElementById('user-menu'),
            usernameDisplay: document.getElementById('username-display'),
            logoutBtn: document.getElementById('logout-btn'),
            authModal: document.getElementById('auth-modal'),
            closeModal: document.getElementById('close-modal'),
            loginTab: document.getElementById('login-tab'),
            registerTab: document.getElementById('register-tab'),
            loginForm: document.getElementById('login-form'),
            registerForm: document.getElementById('register-form'),
            loginError: document.getElementById('login-error'),
            registerError: document.getElementById('register-error'),
            
            // Toast
            toastContainer: document.getElementById('toast-container')
        };
    }
    
    /**
     * Setup event listeners
     */
    setupEventListeners() {
        // Camera controls
        this.elements.startCameraBtn?.addEventListener('click', () => this.startCamera());
        this.elements.stopCameraBtn?.addEventListener('click', () => this.stopCamera());
        
        // Canvas controls
        this.elements.clearCanvasBtn?.addEventListener('click', () => this.clearCanvas());
        this.elements.undoBtn?.addEventListener('click', () => this.undoStroke());
        this.elements.predictBtn?.addEventListener('click', () => this.requestPrediction());
        
        // Auth controls
        this.elements.loginBtn?.addEventListener('click', () => this.showAuthModal());
        this.elements.logoutBtn?.addEventListener('click', () => this.logout());
        this.elements.closeModal?.addEventListener('click', () => this.hideAuthModal());
        this.elements.authModal?.addEventListener('click', (e) => {
            if (e.target === this.elements.authModal) this.hideAuthModal();
        });
        
        // Auth tabs
        this.elements.loginTab?.addEventListener('click', () => this.switchAuthTab('login'));
        this.elements.registerTab?.addEventListener('click', () => this.switchAuthTab('register'));
        
        // Auth forms
        this.elements.loginForm?.addEventListener('submit', (e) => this.handleLogin(e));
        this.elements.registerForm?.addEventListener('submit', (e) => this.handleRegister(e));
        
        // Global events
        window.addEventListener('auth:unauthorized', () => this.handleUnauthorized());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => this.handleKeyboard(e));
    }
    
    /**
     * Initialize modules
     */
    async initializeModules() {
        // Initialize canvas renderer
        if (this.elements.drawingCanvas) {
            canvasRenderer.init(this.elements.drawingCanvas);
            canvasRenderer.onStrokeComplete((stroke) => this.onStrokeComplete(stroke));
        } else {
            console.warn('Drawing canvas not found');
        }
        
        // Initialize video stream
        if (this.elements.videoElement && this.elements.overlayCanvas) {
            try {
                await videoStream.init(this.elements.videoElement, this.elements.overlayCanvas);
                videoStream.onLandmarks((landmarks, handedness) => this.onHandLandmarks(landmarks, handedness));
                console.log('Video stream initialized');
            } catch (error) {
                console.error('Failed to initialize video stream:', error);
                this.showToast('Video initialization failed: ' + error.message, 'warning');
            }
        } else {
            console.warn('Video elements not found:', {
                video: !!this.elements.videoElement,
                overlay: !!this.elements.overlayCanvas
            });
        }
        
        // Initialize WebSocket
        try {
            await wsClient.init();
            wsClient.on('connection', (data) => this.onWebSocketConnection(data));
            wsClient.on('prediction', (data) => this.onPredictionResult(data));
            wsClient.on('error', (data) => this.onWebSocketError(data));
        } catch (error) {
            console.warn('WebSocket initialization failed:', error);
        }
    }
    
    /**
     * Check authentication status
     */
    checkAuth() {
        const token = localStorage.getItem(Config.STORAGE.AUTH_TOKEN);
        const userData = localStorage.getItem(Config.STORAGE.USER_DATA);
        
        if (token && userData) {
            try {
                this.currentUser = JSON.parse(userData);
                this.updateAuthUI(true);
            } catch {
                this.clearAuth();
            }
        } else {
            this.updateAuthUI(false);
        }
    }
    
    /**
     * Check API health
     */
    async checkApiHealth() {
        try {
            const health = await apiClient.checkHealth();
            console.log('API health:', health);
        } catch (error) {
            console.warn('API health check failed:', error);
            this.showToast('Backend server not available', 'warning');
        }
    }
    
    /**
     * Start camera
     */
    async startCamera() {
        try {
            this.elements.startCameraBtn.disabled = true;
            await videoStream.start();
            
            this.isVideoMode = true;
            this.updateVideoStatus(true);
            this.elements.startCameraBtn.classList.add('hidden');
            this.elements.stopCameraBtn.classList.remove('hidden');
            
            this.showToast('Camera started', 'success');
        } catch (error) {
            this.elements.startCameraBtn.disabled = false;
            this.showToast('Failed to start camera: ' + error.message, 'error');
        }
    }
    
    /**
     * Stop camera
     */
    stopCamera() {
        videoStream.stop();
        
        this.isVideoMode = false;
        this.updateVideoStatus(false);
        this.elements.stopCameraBtn.classList.add('hidden');
        this.elements.startCameraBtn.classList.remove('hidden');
        this.elements.startCameraBtn.disabled = false;
    }
    
    /**
     * Update video status indicator
     */
    updateVideoStatus(isConnected) {
        if (isConnected) {
            this.elements.statusIndicator?.classList.add('connected');
            if (this.elements.statusText) {
                this.elements.statusText.textContent = 'Camera active';
            }
        } else {
            this.elements.statusIndicator?.classList.remove('connected');
            if (this.elements.statusText) {
                this.elements.statusText.textContent = 'Camera off';
            }
        }
    }
    
    /**
     * Handle hand landmarks from video
     */
    onHandLandmarks(landmarks, handedness) {
        if (!this.isVideoMode) return;
        
        const isDrawing = videoStream.isDrawingGesture();
        const indexTip = videoStream.getIndexFingerTip();
        
        if (isDrawing && indexTip) {
            // Mirror the x coordinate for natural drawing
            const mirroredX = 1 - indexTip.x;
            
            if (!canvasRenderer.isCurrentlyDrawing()) {
                canvasRenderer.startVideoStroke();
            }
            
            canvasRenderer.addVideoPoint(mirroredX, indexTip.y);
        } else {
            if (canvasRenderer.isCurrentlyDrawing()) {
                const stroke = canvasRenderer.endVideoStroke();
                if (stroke) {
                    this.onStrokeComplete(stroke);
                }
            }
        }
    }
    
    /**
     * Handle stroke completion
     */
    onStrokeComplete(stroke) {
        // Extract features and request prediction
        const features = canvasRenderer.extractStrokeFeatures(stroke);
        
        if (features) {
            // Debounce prediction requests
            clearTimeout(this.predictionDebounce);
            this.predictionDebounce = setTimeout(() => {
                this.sendPrediction(features);
            }, Config.PREDICTION.DEBOUNCE_MS);
        }
    }
    
    /**
     * Send prediction request
     */
    async sendPrediction(features) {
        const startTime = performance.now();
        
        // Try WebSocket first
        if (wsClient.isSocketConnected()) {
            wsClient.requestPrediction(features);
        } else {
            // Fallback to REST API
            try {
                console.log('Sending prediction via REST API');
                const result = await apiClient.predict(features);
                const latency = performance.now() - startTime;
                this.displayPrediction(
                    result.prediction, 
                    result.confidence, 
                    result.alternatives, 
                    result.inference_time || latency
                );
            } catch (error) {
                console.error('Prediction failed:', error);
                this.showToast('Prediction failed: ' + error.message, 'error');
            }
        }
    }
    
    /**
     * Request prediction for current drawing
     */
    async requestPrediction() {
        const lastStroke = canvasRenderer.getLastStroke();
        
        if (!lastStroke || lastStroke.length < Config.FEATURES.MIN_POINTS) {
            this.showToast('Draw a character first', 'info');
            return;
        }
        
        const features = canvasRenderer.extractStrokeFeatures(lastStroke);
        if (features) {
            this.sendPrediction(features);
        }
    }
    
    /**
     * Handle prediction result
     */
    onPredictionResult(data) {
        const { prediction, confidence, alternatives, inference_time } = data;
        this.displayPrediction(prediction, confidence, alternatives, inference_time);
    }
    
    /**
     * Display prediction result
     */
    displayPrediction(prediction, confidence, alternatives = [], inferenceTime = 0) {
        // Hide no prediction message
        this.elements.noPrediction?.classList.add('hidden');
        this.elements.predictionDisplay?.classList.remove('hidden');
        
        // Update main prediction
        if (this.elements.predictedChar) {
            this.elements.predictedChar.textContent = prediction;
        }
        
        // Update confidence
        const confidencePercent = Math.round(confidence * 100);
        if (this.elements.confidenceFill) {
            this.elements.confidenceFill.style.width = `${confidencePercent}%`;
        }
        if (this.elements.confidencePercent) {
            this.elements.confidencePercent.textContent = `${confidencePercent}%`;
        }
        
        // Update alternatives
        if (this.elements.alternativesList) {
            this.elements.alternativesList.innerHTML = alternatives
                .slice(0, 5)
                .map(alt => `
                    <div class="alternative-item">
                        <span class="alternative-char">${alt.char}</span>
                        <span class="alternative-confidence">${Math.round(alt.confidence * 100)}%</span>
                    </div>
                `)
                .join('');
        }
        
        // Update inference time
        if (this.elements.inferenceTime) {
            this.elements.inferenceTime.textContent = `${Math.round(inferenceTime)}ms`;
        }
        
        // Add to history
        this.addToHistory(prediction, confidence);
    }
    
    /**
     * Add prediction to history
     */
    addToHistory(char, confidence) {
        this.predictionHistory.unshift({ char, confidence, timestamp: Date.now() });
        
        // Limit history size
        if (this.predictionHistory.length > Config.PREDICTION.MAX_HISTORY) {
            this.predictionHistory.pop();
        }
        
        this.updateHistoryDisplay();
    }
    
    /**
     * Update history display
     */
    updateHistoryDisplay() {
        if (!this.elements.historyList) return;
        
        if (this.predictionHistory.length === 0) {
            this.elements.historyList.innerHTML = '<span class="empty-history">No predictions yet</span>';
            return;
        }
        
        this.elements.historyList.innerHTML = this.predictionHistory
            .map(item => `
                <span class="history-item">
                    <span class="history-char">${item.char}</span>
                    <span class="history-conf">${Math.round(item.confidence * 100)}%</span>
                </span>
            `)
            .join('');
    }
    
    /**
     * Clear canvas
     */
    clearCanvas() {
        canvasRenderer.clearAll();
        this.resetPredictionDisplay();
    }
    
    /**
     * Undo last stroke
     */
    undoStroke() {
        if (canvasRenderer.undo()) {
            this.showToast('Stroke undone', 'info');
        }
    }
    
    /**
     * Reset prediction display
     */
    resetPredictionDisplay() {
        this.elements.noPrediction?.classList.remove('hidden');
        this.elements.predictionDisplay?.classList.add('hidden');
    }
    
    /**
     * WebSocket connection handler
     */
    onWebSocketConnection(data) {
        if (data.connected) {
            console.log('WebSocket connected');
        } else {
            console.log('WebSocket disconnected:', data.reason);
        }
    }
    
    /**
     * WebSocket error handler
     */
    onWebSocketError(data) {
        console.error('WebSocket error:', data);
        this.showToast('Connection error', 'error');
    }
    
    /**
     * Show auth modal
     */
    showAuthModal() {
        this.elements.authModal?.classList.remove('hidden');
        this.switchAuthTab('login');
    }
    
    /**
     * Hide auth modal
     */
    hideAuthModal() {
        this.elements.authModal?.classList.add('hidden');
        this.clearAuthErrors();
    }
    
    /**
     * Switch auth tab
     */
    switchAuthTab(tab) {
        if (tab === 'login') {
            this.elements.loginTab?.classList.add('active');
            this.elements.registerTab?.classList.remove('active');
            this.elements.loginForm?.classList.remove('hidden');
            this.elements.registerForm?.classList.add('hidden');
        } else {
            this.elements.registerTab?.classList.add('active');
            this.elements.loginTab?.classList.remove('active');
            this.elements.registerForm?.classList.remove('hidden');
            this.elements.loginForm?.classList.add('hidden');
        }
        this.clearAuthErrors();
    }
    
    /**
     * Handle login form submission
     */
    async handleLogin(event) {
        event.preventDefault();
        
        const form = event.target;
        const email = form.querySelector('[name="email"]').value;
        const password = form.querySelector('[name="password"]').value;
        
        try {
            const response = await apiClient.login(email, password);
            this.currentUser = response.user;
            this.updateAuthUI(true);
            this.hideAuthModal();
            this.showToast('Logged in successfully', 'success');
            
            // Update WebSocket auth
            wsClient.updateAuth(response.access_token);
            
        } catch (error) {
            this.showAuthError('login', error.message);
        }
    }
    
    /**
     * Handle register form submission
     */
    async handleRegister(event) {
        event.preventDefault();
        
        const form = event.target;
        const username = form.querySelector('[name="username"]').value;
        const email = form.querySelector('[name="email"]').value;
        const password = form.querySelector('[name="password"]').value;
        
        try {
            const response = await apiClient.register(username, email, password);
            this.showToast('Registration successful! Please login.', 'success');
            this.switchAuthTab('login');
            
        } catch (error) {
            this.showAuthError('register', error.message);
        }
    }
    
    /**
     * Handle logout
     */
    async logout() {
        try {
            await apiClient.logout();
        } catch (error) {
            console.error('Logout error:', error);
        }
        
        this.currentUser = null;
        this.updateAuthUI(false);
        wsClient.updateAuth(null);
        this.showToast('Logged out', 'info');
    }
    
    /**
     * Handle unauthorized response
     */
    handleUnauthorized() {
        this.currentUser = null;
        this.updateAuthUI(false);
        this.showToast('Session expired. Please login again.', 'warning');
    }
    
    /**
     * Update auth UI
     */
    updateAuthUI(isAuthenticated) {
        if (isAuthenticated && this.currentUser) {
            this.elements.loginBtn?.classList.add('hidden');
            this.elements.userMenu?.classList.remove('hidden');
            if (this.elements.usernameDisplay) {
                this.elements.usernameDisplay.textContent = this.currentUser.username;
            }
        } else {
            this.elements.loginBtn?.classList.remove('hidden');
            this.elements.userMenu?.classList.add('hidden');
        }
    }
    
    /**
     * Clear auth
     */
    clearAuth() {
        localStorage.removeItem(Config.STORAGE.AUTH_TOKEN);
        localStorage.removeItem(Config.STORAGE.USER_DATA);
        this.currentUser = null;
    }
    
    /**
     * Show auth error
     */
    showAuthError(type, message) {
        const errorElement = type === 'login' ? this.elements.loginError : this.elements.registerError;
        if (errorElement) {
            errorElement.textContent = message;
            errorElement.classList.remove('hidden');
        }
    }
    
    /**
     * Clear auth errors
     */
    clearAuthErrors() {
        this.elements.loginError?.classList.add('hidden');
        this.elements.registerError?.classList.add('hidden');
    }
    
    /**
     * Show toast notification
     */
    showToast(message, type = 'info') {
        if (!this.elements.toastContainer) return;
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        
        this.elements.toastContainer.appendChild(toast);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.style.animation = 'slideIn var(--transition-normal) reverse';
            setTimeout(() => toast.remove(), 250);
        }, 3000);
    }
    
    /**
     * Handle keyboard shortcuts
     */
    handleKeyboard(event) {
        // Ignore if in input field
        if (event.target.tagName === 'INPUT') return;
        
        switch (event.key) {
            case 'c':
                if (event.ctrlKey || event.metaKey) {
                    event.preventDefault();
                    this.clearCanvas();
                }
                break;
            case 'z':
                if (event.ctrlKey || event.metaKey) {
                    event.preventDefault();
                    this.undoStroke();
                }
                break;
            case 'Enter':
                if (event.ctrlKey || event.metaKey) {
                    event.preventDefault();
                    this.requestPrediction();
                }
                break;
            case 'Escape':
                this.hideAuthModal();
                break;
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.init();
    
    // Expose to window for debugging
    if (Config.isDevelopment()) {
        window.app = app;
        window.apiClient = apiClient;
        window.videoStream = videoStream;
        window.canvasRenderer = canvasRenderer;
        window.wsClient = wsClient;
    }
});

export default App;
