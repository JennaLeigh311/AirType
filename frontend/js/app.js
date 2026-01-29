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
        
        // Correction state
        this.currentPredictionData = null;
        this.selectedCorrection = null;
        this.contributionTotal = 0;
        
        // Training state
        this.trainingStats = null;
        this.isTraining = false;
        
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
            
            // Initialize correction grids
            this.initCorrectionGrids();
            
            // Load contribution count
            this.contributionTotal = parseInt(localStorage.getItem('airtype_contribution_count') || '0');
            if (this.elements.contributionCount) {
                this.elements.contributionCount.textContent = this.contributionTotal;
            }
            
            // Check authentication
            this.checkAuth();
            
            // Check API health
            this.checkApiHealth();
            
            // Load training stats
            this.loadTrainingStats();
            
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
            toastContainer: document.getElementById('toast-container'),
            
            // Correction Modal
            correctionModal: document.getElementById('correction-modal'),
            closeCorrectionBtn: document.getElementById('close-correction'),
            correctionPredicted: document.getElementById('correction-predicted'),
            correctionConfidence: document.getElementById('correction-confidence'),
            lowercaseGrid: document.getElementById('lowercase-grid'),
            uppercaseGrid: document.getElementById('uppercase-grid'),
            numbersGrid: document.getElementById('numbers-grid'),
            manualCharInput: document.getElementById('manual-char-input'),
            confirmCorrectBtn: document.getElementById('confirm-correct'),
            submitCorrectionBtn: document.getElementById('submit-correction'),
            skipCorrectionBtn: document.getElementById('skip-correction'),
            contributionCount: document.getElementById('contribution-count'),
            
            // Training Panel
            totalSamples: document.getElementById('total-samples'),
            overallAccuracy: document.getElementById('overall-accuracy'),
            practiceSuggestion: document.getElementById('practice-suggestion'),
            suggestedChars: document.getElementById('suggested-chars'),
            getSuggestionsBtn: document.getElementById('get-suggestions-btn'),
            trainModelBtn: document.getElementById('train-model-btn'),
            trainingStatus: document.getElementById('training-status'),
            trainingStatusText: document.getElementById('training-status-text')
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
        
        // Correction modal controls
        this.elements.closeCorrectionBtn?.addEventListener('click', () => this.hideCorrectionModal());
        this.elements.correctionModal?.addEventListener('click', (e) => {
            if (e.target === this.elements.correctionModal) this.hideCorrectionModal();
        });
        this.elements.confirmCorrectBtn?.addEventListener('click', () => this.confirmCorrect());
        this.elements.submitCorrectionBtn?.addEventListener('click', () => this.submitCorrection());
        this.elements.skipCorrectionBtn?.addEventListener('click', () => this.hideCorrectionModal());
        this.elements.manualCharInput?.addEventListener('input', (e) => this.handleManualInput(e));
        
        // Training panel controls
        this.elements.getSuggestionsBtn?.addEventListener('click', () => this.loadPracticeSuggestions());
        this.elements.trainModelBtn?.addEventListener('click', () => this.trainModel());
        
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
        
        // Initialize correction modal character grids
        this.initCorrectionGrids();
        
        // Load contribution count from localStorage
        this.contributionTotal = parseInt(localStorage.getItem('airtype_contribution_count') || '0');
        if (this.elements.contributionCount) {
            this.elements.contributionCount.textContent = this.contributionTotal;
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
        
        console.log('=== SENDING PREDICTION ===');
        console.log('Features count:', features.length, 'rows');
        
        // Use REST API for manual predictions (WebSocket can be used later for real-time)
        try {
            console.log('Sending prediction via REST API to', Config.API_BASE_URL + '/predictions/predict');
            const result = await apiClient.predict(features);
            console.log('Prediction result received:', result);
            const latency = performance.now() - startTime;
            
            // Display prediction and show correction modal
            this.displayPrediction(
                result.prediction, 
                result.confidence, 
                result.alternatives, 
                result.inference_time || latency,
                features, // Pass features for feedback
                result.model_version
            );
            console.log('Prediction displayed successfully');
        } catch (error) {
            console.error('=== PREDICTION FAILED ===');
            console.error('Error:', error);
            this.showToast('Prediction failed: ' + error.message, 'error');
        }
    }
    
    /**
     * Request prediction for current drawing
     */
    async requestPrediction() {
        console.log('\n========================================');
        console.log('=== REQUEST PREDICTION BUTTON CLICKED ==');
        console.log('========================================');
        
        // Try to get the last completed stroke first
        let stroke = canvasRenderer.getLastStroke();
        console.log('Step 1 - getLastStroke():', stroke ? `Got ${stroke.length} points` : 'EMPTY/NULL');
        
        // If no completed stroke, try current stroke being drawn
        if (!stroke || stroke.length < Config.FEATURES.MIN_POINTS) {
            console.log('Step 1 failed - trying getCurrentStroke()...');
            stroke = canvasRenderer.getCurrentStroke();
            console.log('Step 2 - getCurrentStroke():', stroke ? `Got ${stroke.length} points` : 'EMPTY/NULL');
        }
        
        // Also check all strokes
        const allStrokes = canvasRenderer.getStrokes();
        console.log('Step 3 - All strokes in memory:', allStrokes.length);
        if (allStrokes.length > 0) {
            console.log('Strokes details:', allStrokes.map((s, i) => `Stroke ${i}: ${s.length} points`));
        }
        
        // If still no stroke, try the last from allStrokes directly
        if ((!stroke || stroke.length < Config.FEATURES.MIN_POINTS) && allStrokes.length > 0) {
            stroke = allStrokes[allStrokes.length - 1];
            console.log('Step 4 - Using last stroke from allStrokes:', stroke.length, 'points');
        }
        
        if (!stroke || stroke.length < Config.FEATURES.MIN_POINTS) {
            console.log('FAILED: No valid stroke found. MIN_POINTS required:', Config.FEATURES.MIN_POINTS);
            this.showToast('Draw a character first (need at least ' + Config.FEATURES.MIN_POINTS + ' points)', 'info');
            return;
        }
        
        console.log('Step 5 - Valid stroke found with', stroke.length, 'points');
        console.log('Stroke data sample (first 3 points):', stroke.slice(0, 3));
        
        const features = canvasRenderer.extractStrokeFeatures(stroke);
        console.log('Step 6 - Extracted features:', features ? `${features.length} rows` : 'NULL - extraction failed!');
        
        if (features) {
            console.log('Step 7 - Sending prediction request...');
            await this.sendPrediction(features);
        } else {
            console.error('Feature extraction returned null!');
            this.showToast('Failed to extract features from stroke', 'error');
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
    displayPrediction(prediction, confidence, alternatives = [], inferenceTime = 0, features = null, modelVersion = '1.0.0') {
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
        
        // Show correction modal if features are provided
        if (features) {
            this.showCorrectionModal({
                prediction,
                confidence,
                alternatives,
                inference_time: inferenceTime,
                features,
                model_version: modelVersion
            });
        }
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
     * Initialize correction character grids
     */
    initCorrectionGrids() {
        // Lowercase letters (a-z)
        if (this.elements.lowercaseGrid) {
            for (let i = 97; i <= 122; i++) {
                const char = String.fromCharCode(i);
                const btn = document.createElement('button');
                btn.className = 'char-btn';
                btn.textContent = char;
                btn.addEventListener('click', () => this.selectCharacter(char));
                this.elements.lowercaseGrid.appendChild(btn);
            }
        }
        
        // Uppercase letters (A-Z)
        if (this.elements.uppercaseGrid) {
            for (let i = 65; i <= 90; i++) {
                const char = String.fromCharCode(i);
                const btn = document.createElement('button');
                btn.className = 'char-btn';
                btn.textContent = char;
                btn.addEventListener('click', () => this.selectCharacter(char));
                this.elements.uppercaseGrid.appendChild(btn);
            }
        }
        
        // Numbers (0-9)
        if (this.elements.numbersGrid) {
            for (let i = 0; i <= 9; i++) {
                const char = i.toString();
                const btn = document.createElement('button');
                btn.className = 'char-btn';
                btn.textContent = char;
                btn.addEventListener('click', () => this.selectCharacter(char));
                this.elements.numbersGrid.appendChild(btn);
            }
        }
    }
    
    /**
     * Show correction modal after prediction
     */
    showCorrectionModal(predictionData) {
        this.currentPredictionData = predictionData;
        this.selectedCorrection = null;
        
        // Update modal content
        if (this.elements.correctionPredicted) {
            this.elements.correctionPredicted.textContent = predictionData.prediction;
        }
        if (this.elements.correctionConfidence) {
            this.elements.correctionConfidence.textContent = `${Math.round(predictionData.confidence * 100)}%`;
        }
        
        // Reset selection
        document.querySelectorAll('.char-btn').forEach(btn => btn.classList.remove('selected'));
        if (this.elements.manualCharInput) {
            this.elements.manualCharInput.value = '';
        }
        if (this.elements.submitCorrectionBtn) {
            this.elements.submitCorrectionBtn.disabled = true;
        }
        
        // Show modal
        if (this.elements.correctionModal) {
            this.elements.correctionModal.classList.remove('hidden');
        }
    }
    
    /**
     * Hide correction modal
     */
    hideCorrectionModal() {
        if (this.elements.correctionModal) {
            this.elements.correctionModal.classList.add('hidden');
        }
        this.currentPredictionData = null;
        this.selectedCorrection = null;
    }
    
    /**
     * Select a character from the grid
     */
    selectCharacter(char) {
        this.selectedCorrection = char;
        
        // Update UI
        document.querySelectorAll('.char-btn').forEach(btn => {
            if (btn.textContent === char) {
                btn.classList.add('selected');
            } else {
                btn.classList.remove('selected');
            }
        });
        
        if (this.elements.manualCharInput) {
            this.elements.manualCharInput.value = char;
        }
        if (this.elements.submitCorrectionBtn) {
            this.elements.submitCorrectionBtn.disabled = false;
        }
    }
    
    /**
     * Handle manual character input
     */
    handleManualInput(event) {
        const value = event.target.value.toUpperCase();
        
        if (value && /^[a-zA-Z0-9]$/.test(value)) {
            this.selectCharacter(value);
        } else {
            this.selectedCorrection = null;
            document.querySelectorAll('.char-btn').forEach(btn => btn.classList.remove('selected'));
            if (this.elements.submitCorrectionBtn) {
                this.elements.submitCorrectionBtn.disabled = true;
            }
        }
    }
    
    /**
     * Confirm prediction was correct
     */
    async confirmCorrect() {
        if (!this.currentPredictionData) return;
        
        await this.sendFeedback(
            this.currentPredictionData.prediction,
            'confirmed'
        );
        
        this.showToast('Thanks for confirming! ✓', 'success');
        this.hideCorrectionModal();
    }
    
    /**
     * Submit correction
     */
    async submitCorrection() {
        if (!this.selectedCorrection || !this.currentPredictionData) return;
        
        await this.sendFeedback(
            this.selectedCorrection,
            'corrected'
        );
        
        this.showToast(`Correction submitted: ${this.selectedCorrection} ✓`, 'success');
        this.hideCorrectionModal();
    }
    
    /**
     * Send feedback to backend
     */
    async sendFeedback(actualChar, correctionType) {
        try {
            console.log('Sending feedback:', {
                predicted: this.currentPredictionData.prediction,
                actual: actualChar,
                type: correctionType
            });
            
            const response = await fetch(Config.API_BASE_URL + '/training/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    stroke_features: this.currentPredictionData.features,
                    predicted_char: this.currentPredictionData.prediction,
                    predicted_confidence: this.currentPredictionData.confidence,
                    alternatives: this.currentPredictionData.alternatives,
                    actual_char: actualChar,
                    correction_type: correctionType,
                    model_version: this.currentPredictionData.model_version || '1.0.0',
                    inference_time_ms: this.currentPredictionData.inference_time || 0
                })
            });
            
            if (response.ok) {
                // Update contribution count
                this.contributionTotal++;
                localStorage.setItem('airtype_contribution_count', this.contributionTotal.toString());
                if (this.elements.contributionCount) {
                    this.elements.contributionCount.textContent = this.contributionTotal;
                }
                
                console.log('Feedback sent successfully');
                
                // Refresh training stats
                this.loadTrainingStats();
            } else {
                console.error('Failed to send feedback:', await response.text());
            }
        } catch (error) {
            console.error('Error sending feedback:', error);
        }
    }
    
    /**
     * Load training statistics
     */
    async loadTrainingStats() {
        try {
            const response = await fetch(Config.API_BASE_URL + '/training/stats');
            if (response.ok) {
                this.trainingStats = await response.json();
                this.updateTrainingPanel();
            }
        } catch (error) {
            console.error('Failed to load training stats:', error);
        }
    }
    
    /**
     * Update training panel UI
     */
    updateTrainingPanel() {
        if (!this.trainingStats) return;
        
        if (this.elements.totalSamples) {
            this.elements.totalSamples.textContent = this.trainingStats.total_samples || 0;
        }
        
        if (this.elements.overallAccuracy) {
            const acc = this.trainingStats.overall_accuracy;
            this.elements.overallAccuracy.textContent = acc ? `${acc}%` : '-';
        }
        
        // Enable train button if enough samples
        if (this.elements.trainModelBtn) {
            const samples = this.trainingStats.total_samples || 0;
            this.elements.trainModelBtn.disabled = samples < 50;
            this.elements.trainModelBtn.textContent = samples < 50 
                ? `Train Model (${samples}/50 samples)`
                : `Train Model (${samples} samples)`;
        }
    }
    
    /**
     * Load practice suggestions from active learning
     */
    async loadPracticeSuggestions() {
        try {
            this.elements.getSuggestionsBtn.disabled = true;
            this.elements.getSuggestionsBtn.textContent = 'Loading...';
            
            const response = await fetch(Config.API_BASE_URL + '/training/suggestions');
            if (response.ok) {
                const data = await response.json();
                this.displayPracticeSuggestions(data);
            } else {
                this.showToast('Failed to load suggestions', 'error');
            }
        } catch (error) {
            console.error('Failed to load suggestions:', error);
            this.showToast('Failed to load suggestions', 'error');
        } finally {
            this.elements.getSuggestionsBtn.disabled = false;
            this.elements.getSuggestionsBtn.textContent = 'Get Practice Suggestions';
        }
    }
    
    /**
     * Display practice suggestions
     */
    displayPracticeSuggestions(data) {
        if (!this.elements.practiceSuggestion || !this.elements.suggestedChars) return;
        
        this.elements.practiceSuggestion.classList.remove('hidden');
        this.elements.suggestedChars.innerHTML = '';
        
        // Display top suggested characters
        const suggestions = data.suggestions || [];
        suggestions.slice(0, 10).forEach(suggestion => {
            const charEl = document.createElement('div');
            charEl.className = 'suggested-char';
            charEl.innerHTML = `
                <span class="char">${suggestion.char}</span>
                <span class="priority">★${Math.round(suggestion.score)}</span>
            `;
            charEl.title = suggestion.reasons.join('\n');
            this.elements.suggestedChars.appendChild(charEl);
        });
        
        // Show practice session
        if (data.practice_session) {
            const sessionEl = document.createElement('div');
            sessionEl.className = 'practice-session';
            sessionEl.innerHTML = `
                <p style="margin-top: 0.5rem; font-size: 0.75rem; color: var(--color-text-muted);">
                    Practice session: ${data.practice_session.slice(0, 15).join(' ')}...
                </p>
            `;
            this.elements.suggestedChars.appendChild(sessionEl);
        }
    }
    
    /**
     * Train the model with collected data
     */
    async trainModel() {
        if (this.isTraining) return;
        
        try {
            this.isTraining = true;
            this.elements.trainModelBtn.disabled = true;
            this.elements.trainingStatus.classList.remove('hidden', 'error', 'success');
            this.elements.trainingStatusText.textContent = 'Training model... This may take a minute.';
            
            const response = await fetch(Config.API_BASE_URL + '/training/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    min_samples: 50,
                    epochs: 10,
                    augment: true,
                    augmentation_factor: 5
                })
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.elements.trainingStatus.classList.add('success');
                const accuracy = result.results?.final_val_accuracy || result.results?.final_train_accuracy || 0;
                this.elements.trainingStatusText.textContent = 
                    `✓ Training complete! Accuracy: ${accuracy}% (${result.results?.training_time_seconds}s)`;
                this.showToast('Model trained successfully!', 'success');
                
                // Refresh stats
                this.loadTrainingStats();
            } else {
                this.elements.trainingStatus.classList.add('error');
                this.elements.trainingStatusText.textContent = `✗ ${result.error}`;
                this.showToast(result.error, 'error');
            }
        } catch (error) {
            console.error('Training failed:', error);
            this.elements.trainingStatus.classList.add('error');
            this.elements.trainingStatusText.textContent = `✗ Training failed: ${error.message}`;
            this.showToast('Training failed', 'error');
        } finally {
            this.isTraining = false;
            this.elements.trainModelBtn.disabled = (this.trainingStats?.total_samples || 0) < 50;
        }
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
