/**
 * AirType - WebSocket Client Module
 * Handles real-time communication with the backend
 */

import Config from './config.js';

class WebSocketClient {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.reconnectTimeout = null;
        this.pingInterval = null;
        this.eventHandlers = new Map();
        this.sessionId = null;
    }
    
    /**
     * Initialize WebSocket connection
     */
    async init() {
        return new Promise((resolve, reject) => {
            try {
                // Load Socket.IO client library
                if (!window.io) {
                    const script = document.createElement('script');
                    script.src = 'https://cdn.socket.io/4.6.0/socket.io.min.js';
                    script.onload = () => {
                        this.connect();
                        resolve();
                    };
                    script.onerror = () => reject(new Error('Failed to load Socket.IO'));
                    document.head.appendChild(script);
                } else {
                    this.connect();
                    resolve();
                }
            } catch (error) {
                reject(error);
            }
        });
    }
    
    /**
     * Connect to WebSocket server
     */
    connect() {
        if (this.socket && this.isConnected) return;
        
        const token = localStorage.getItem(Config.STORAGE.AUTH_TOKEN);
        
        this.socket = io(Config.WS_URL, {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionDelay: Config.WEBSOCKET.RECONNECT_INTERVAL,
            reconnectionAttempts: Config.WEBSOCKET.MAX_RECONNECT_ATTEMPTS,
            auth: token ? { token } : {}
        });
        
        this.setupEventListeners();
    }
    
    /**
     * Setup WebSocket event listeners
     */
    setupEventListeners() {
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.startPing();
            this.emit('connection', { connected: true });
        });
        
        this.socket.on('disconnect', (reason) => {
            console.log('WebSocket disconnected:', reason);
            this.isConnected = false;
            this.stopPing();
            this.emit('connection', { connected: false, reason });
        });
        
        this.socket.on('connect_error', (error) => {
            console.error('WebSocket connection error:', error);
            this.emit('error', { type: 'connection', error: error.message });
        });
        
        // Custom events
        this.socket.on('prediction_result', (data) => {
            this.emit('prediction', data);
        });
        
        this.socket.on('session_created', (data) => {
            this.sessionId = data.session_id;
            this.emit('session', { type: 'created', ...data });
        });
        
        this.socket.on('session_completed', (data) => {
            this.emit('session', { type: 'completed', ...data });
        });
        
        this.socket.on('error', (data) => {
            this.emit('error', data);
        });
        
        this.socket.on('pong', () => {
            // Server responded to ping
        });
    }
    
    /**
     * Start ping interval
     */
    startPing() {
        this.pingInterval = setInterval(() => {
            if (this.isConnected) {
                this.socket.emit('ping');
            }
        }, Config.WEBSOCKET.PING_INTERVAL);
    }
    
    /**
     * Stop ping interval
     */
    stopPing() {
        if (this.pingInterval) {
            clearInterval(this.pingInterval);
            this.pingInterval = null;
        }
    }
    
    /**
     * Disconnect from WebSocket server
     */
    disconnect() {
        this.stopPing();
        
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
        
        this.isConnected = false;
        this.sessionId = null;
    }
    
    /**
     * Send stroke data for prediction
     */
    sendStroke(strokeData) {
        if (!this.isConnected) {
            console.warn('WebSocket not connected');
            return false;
        }
        
        this.socket.emit('stroke_data', {
            session_id: this.sessionId,
            stroke: strokeData,
            timestamp: Date.now()
        });
        
        return true;
    }
    
    /**
     * Send frame data for processing
     */
    sendFrame(frameData) {
        if (!this.isConnected) return false;
        
        this.socket.emit('frame_data', {
            session_id: this.sessionId,
            frame: frameData,
            timestamp: Date.now()
        });
        
        return true;
    }
    
    /**
     * Start new session
     */
    startSession(deviceInfo = {}) {
        if (!this.isConnected) return false;
        
        this.socket.emit('start_session', {
            device_info: deviceInfo,
            timestamp: Date.now()
        });
        
        return true;
    }
    
    /**
     * End current session
     */
    endSession() {
        if (!this.isConnected || !this.sessionId) return false;
        
        this.socket.emit('end_session', {
            session_id: this.sessionId,
            timestamp: Date.now()
        });
        
        this.sessionId = null;
        return true;
    }
    
    /**
     * Send prediction request
     */
    requestPrediction(features) {
        if (!this.isConnected) return false;
        
        this.socket.emit('predict', {
            session_id: this.sessionId,
            features: features,
            timestamp: Date.now()
        });
        
        return true;
    }
    
    /**
     * Register event handler
     */
    on(event, handler) {
        if (!this.eventHandlers.has(event)) {
            this.eventHandlers.set(event, []);
        }
        this.eventHandlers.get(event).push(handler);
    }
    
    /**
     * Remove event handler
     */
    off(event, handler) {
        if (!this.eventHandlers.has(event)) return;
        
        const handlers = this.eventHandlers.get(event);
        const index = handlers.indexOf(handler);
        
        if (index !== -1) {
            handlers.splice(index, 1);
        }
    }
    
    /**
     * Emit event to registered handlers
     */
    emit(event, data) {
        if (!this.eventHandlers.has(event)) return;
        
        for (const handler of this.eventHandlers.get(event)) {
            try {
                handler(data);
            } catch (error) {
                console.error(`Error in event handler for ${event}:`, error);
            }
        }
    }
    
    /**
     * Check if connected
     */
    isSocketConnected() {
        return this.isConnected;
    }
    
    /**
     * Get current session ID
     */
    getSessionId() {
        return this.sessionId;
    }
    
    /**
     * Update authentication
     */
    updateAuth(token) {
        if (this.socket) {
            this.socket.auth = token ? { token } : {};
            
            // Reconnect with new auth
            if (this.isConnected) {
                this.socket.disconnect();
                this.socket.connect();
            }
        }
    }
}

// Export singleton instance
const wsClient = new WebSocketClient();
export default wsClient;
