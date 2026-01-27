/**
 * AirType - Video Stream Module
 * Handles webcam capture and MediaPipe hand tracking
 */

import Config from './config.js';

class VideoStream {
    constructor() {
        this.videoElement = null;
        this.overlayCanvas = null;
        this.overlayCtx = null;
        this.stream = null;
        this.hands = null;
        this.isRunning = false;
        this.handLandmarks = null;
        this.onLandmarksCallback = null;
        this.animationFrameId = null;
        
        // Kalman filter state for smoothing
        this.kalmanState = null;
        this.initKalmanFilter();
    }
    
    /**
     * Initialize Kalman filter for coordinate smoothing
     */
    initKalmanFilter() {
        this.kalmanState = {
            x: { estimate: 0, error: 1, measurementNoise: 0.1, processNoise: 0.01 },
            y: { estimate: 0, error: 1, measurementNoise: 0.1, processNoise: 0.01 },
            z: { estimate: 0, error: 1, measurementNoise: 0.1, processNoise: 0.01 }
        };
    }
    
    /**
     * Apply Kalman filter to measurement
     */
    kalmanFilter(measurement, state) {
        // Prediction
        const predictedEstimate = state.estimate;
        const predictedError = state.error + state.processNoise;
        
        // Update
        const kalmanGain = predictedError / (predictedError + state.measurementNoise);
        state.estimate = predictedEstimate + kalmanGain * (measurement - predictedEstimate);
        state.error = (1 - kalmanGain) * predictedError;
        
        return state.estimate;
    }
    
    /**
     * Initialize video stream and hand tracking
     */
    async init(videoElement, overlayCanvas) {
        this.videoElement = videoElement;
        this.overlayCanvas = overlayCanvas;
        this.overlayCtx = overlayCanvas.getContext('2d');
        
        // Load MediaPipe Hands
        await this.loadMediaPipeHands();
    }
    
    /**
     * Load MediaPipe Hands library
     */
    async loadMediaPipeHands() {
        return new Promise((resolve, reject) => {
            // Check if MediaPipe is already loaded
            if (window.Hands) {
                this.initializeHands();
                resolve();
                return;
            }
            
            // Load the script dynamically
            const script = document.createElement('script');
            script.src = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js';
            script.onload = () => {
                this.initializeHands();
                resolve();
            };
            script.onerror = () => reject(new Error('Failed to load MediaPipe Hands'));
            document.head.appendChild(script);
        });
    }
    
    /**
     * Initialize MediaPipe Hands instance
     */
    initializeHands() {
        this.hands = new window.Hands({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
            }
        });
        
        this.hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.7
        });
        
        this.hands.onResults((results) => this.onHandResults(results));
    }
    
    /**
     * Start video stream
     */
    async start() {
        if (this.isRunning) return;
        
        // Check if mediaDevices is available
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera access not supported. Please use HTTPS or localhost.');
        }
        
        // Ensure video and canvas elements are set
        if (!this.videoElement || !this.overlayCanvas) {
            throw new Error('Video elements not initialized. Call init() first.');
        }
        
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: Config.CAMERA.WIDTH },
                    height: { ideal: Config.CAMERA.HEIGHT },
                    frameRate: { ideal: Config.CAMERA.FRAME_RATE },
                    facingMode: Config.CAMERA.FACING_MODE
                }
            });
            
            this.videoElement.srcObject = this.stream;
            await this.videoElement.play();
            
            // Set canvas size to match video
            this.overlayCanvas.width = this.videoElement.videoWidth || Config.CAMERA.WIDTH;
            this.overlayCanvas.height = this.videoElement.videoHeight || Config.CAMERA.HEIGHT;
            
            this.isRunning = true;
            this.processFrame();
            
            return true;
        } catch (error) {
            console.error('Failed to start video stream:', error);
            throw error;
        }
    }
    
    /**
     * Stop video stream
     */
    stop() {
        this.isRunning = false;
        
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
            this.animationFrameId = null;
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.videoElement.srcObject = null;
        this.clearOverlay();
        this.handLandmarks = null;
    }
    
    /**
     * Process video frame
     */
    async processFrame() {
        if (!this.isRunning) return;
        
        if (this.videoElement.readyState >= 2) {
            await this.hands.send({ image: this.videoElement });
        }
        
        this.animationFrameId = requestAnimationFrame(() => this.processFrame());
    }
    
    /**
     * Handle hand detection results
     */
    onHandResults(results) {
        this.clearOverlay();
        
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];
            
            // Smooth landmarks using Kalman filter
            const smoothedLandmarks = this.smoothLandmarks(landmarks);
            
            // Draw hand landmarks
            this.drawHandLandmarks(smoothedLandmarks);
            
            // Store landmarks and trigger callback
            this.handLandmarks = smoothedLandmarks;
            
            if (this.onLandmarksCallback) {
                this.onLandmarksCallback(smoothedLandmarks, results.multiHandedness[0]);
            }
        } else {
            this.handLandmarks = null;
        }
    }
    
    /**
     * Smooth landmarks using Kalman filter
     */
    smoothLandmarks(landmarks) {
        return landmarks.map((landmark, index) => {
            // Only apply heavy smoothing to index finger tip (landmark 8)
            if (index === 8) {
                return {
                    x: this.kalmanFilter(landmark.x, this.kalmanState.x),
                    y: this.kalmanFilter(landmark.y, this.kalmanState.y),
                    z: this.kalmanFilter(landmark.z, this.kalmanState.z)
                };
            }
            return { ...landmark };
        });
    }
    
    /**
     * Draw hand landmarks on overlay canvas
     */
    drawHandLandmarks(landmarks) {
        const ctx = this.overlayCtx;
        const width = this.overlayCanvas.width;
        const height = this.overlayCanvas.height;
        
        // Define hand connections
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4],         // Thumb
            [0, 5], [5, 6], [6, 7], [7, 8],         // Index finger
            [0, 9], [9, 10], [10, 11], [11, 12],    // Middle finger
            [0, 13], [13, 14], [14, 15], [15, 16],  // Ring finger
            [0, 17], [17, 18], [18, 19], [19, 20],  // Pinky
            [5, 9], [9, 13], [13, 17]               // Palm
        ];
        
        // Draw connections
        ctx.strokeStyle = Config.CANVAS.CONNECTION_COLOR;
        ctx.lineWidth = Config.CANVAS.CONNECTION_WIDTH;
        
        for (const [start, end] of connections) {
            const startPoint = landmarks[start];
            const endPoint = landmarks[end];
            
            ctx.beginPath();
            ctx.moveTo(startPoint.x * width, startPoint.y * height);
            ctx.lineTo(endPoint.x * width, endPoint.y * height);
            ctx.stroke();
        }
        
        // Draw landmarks
        for (let i = 0; i < landmarks.length; i++) {
            const landmark = landmarks[i];
            const x = landmark.x * width;
            const y = landmark.y * height;
            
            ctx.beginPath();
            ctx.arc(x, y, Config.CANVAS.LANDMARK_SIZE, 0, 2 * Math.PI);
            
            // Highlight index finger tip
            if (i === 8) {
                ctx.fillStyle = Config.CANVAS.LANDMARK_COLOR;
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.fill();
                ctx.stroke();
            } else {
                ctx.fillStyle = Config.CANVAS.HAND_COLOR;
                ctx.fill();
            }
        }
        
        // Draw index finger tip position indicator
        const indexTip = landmarks[8];
        const tipX = indexTip.x * width;
        const tipY = indexTip.y * height;
        
        // Draw crosshair
        ctx.strokeStyle = Config.CANVAS.LANDMARK_COLOR;
        ctx.lineWidth = 1;
        
        ctx.beginPath();
        ctx.moveTo(tipX - 15, tipY);
        ctx.lineTo(tipX - 5, tipY);
        ctx.moveTo(tipX + 5, tipY);
        ctx.lineTo(tipX + 15, tipY);
        ctx.moveTo(tipX, tipY - 15);
        ctx.lineTo(tipX, tipY - 5);
        ctx.moveTo(tipX, tipY + 5);
        ctx.lineTo(tipX, tipY + 15);
        ctx.stroke();
    }
    
    /**
     * Clear overlay canvas
     */
    clearOverlay() {
        this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
    }
    
    /**
     * Set callback for landmark updates
     */
    onLandmarks(callback) {
        this.onLandmarksCallback = callback;
    }
    
    /**
     * Get current hand landmarks
     */
    getLandmarks() {
        return this.handLandmarks;
    }
    
    /**
     * Get index finger tip position
     */
    getIndexFingerTip() {
        if (!this.handLandmarks) return null;
        
        return {
            x: this.handLandmarks[8].x,
            y: this.handLandmarks[8].y,
            z: this.handLandmarks[8].z
        };
    }
    
    /**
     * Check if drawing gesture is detected
     * Drawing when index finger is extended and middle finger is not
     */
    isDrawingGesture() {
        if (!this.handLandmarks) return false;
        
        const landmarks = this.handLandmarks;
        
        // Index finger tip (8) should be higher than index finger PIP (6)
        const indexExtended = landmarks[8].y < landmarks[6].y;
        
        // Middle finger tip (12) should be lower than middle finger PIP (10)
        const middleRetracted = landmarks[12].y > landmarks[10].y;
        
        // Ring finger tip (16) should be lower than ring finger PIP (14)
        const ringRetracted = landmarks[16].y > landmarks[14].y;
        
        return indexExtended && middleRetracted && ringRetracted;
    }
    
    /**
     * Check if stream is running
     */
    isStreamRunning() {
        return this.isRunning;
    }
}

// Export singleton instance
const videoStream = new VideoStream();
export default videoStream;
