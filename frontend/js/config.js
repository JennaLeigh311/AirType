/**
 * AirType - Configuration Module
 * Application-wide configuration settings
 */

const Config = {
    // API Settings
    API_BASE_URL: 'http://localhost:5001/api',
    WS_URL: 'ws://localhost:5001',
    
    // Camera Settings
    CAMERA: {
        WIDTH: 640,
        HEIGHT: 480,
        FRAME_RATE: 30,
        FACING_MODE: 'user'
    },
    
    // Canvas Settings
    CANVAS: {
        STROKE_COLOR: '#3b82f6',
        STROKE_WIDTH: 3,
        HAND_COLOR: '#22c55e',
        LANDMARK_COLOR: '#f59e0b',
        LANDMARK_SIZE: 5,
        CONNECTION_COLOR: 'rgba(34, 197, 94, 0.5)',
        CONNECTION_WIDTH: 2
    },
    
    // Prediction Settings
    PREDICTION: {
        MIN_CONFIDENCE: 0.5,
        DEBOUNCE_MS: 300,
        MAX_HISTORY: 20
    },
    
    // WebSocket Settings
    WEBSOCKET: {
        RECONNECT_INTERVAL: 3000,
        MAX_RECONNECT_ATTEMPTS: 5,
        PING_INTERVAL: 30000
    },
    
    // Feature Extraction Settings
    FEATURES: {
        MIN_POINTS: 10,
        SMOOTHING_WINDOW: 5,
        VELOCITY_THRESHOLD: 0.01
    },
    
    // Storage Keys
    STORAGE: {
        AUTH_TOKEN: 'airtype_auth_token',
        USER_DATA: 'airtype_user_data',
        SETTINGS: 'airtype_settings'
    },
    
    // Classes (a-z, A-Z, 0-9)
    CLASSES: [
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
        'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
    ],
    
    // Get class by index
    getClassByIndex(index) {
        return this.CLASSES[index] || null;
    },
    
    // Get index by class
    getIndexByClass(char) {
        return this.CLASSES.indexOf(char);
    },
    
    // Get full API URL
    getApiUrl(endpoint) {
        return `${this.API_BASE_URL}${endpoint}`;
    },
    
    // Check if in development mode
    isDevelopment() {
        return window.location.hostname === 'localhost' || 
               window.location.hostname === '127.0.0.1';
    }
};

// Freeze config to prevent modifications
Object.freeze(Config);
Object.freeze(Config.CAMERA);
Object.freeze(Config.CANVAS);
Object.freeze(Config.PREDICTION);
Object.freeze(Config.WEBSOCKET);
Object.freeze(Config.FEATURES);
Object.freeze(Config.STORAGE);
Object.freeze(Config.CLASSES);

export default Config;
