/**
 * AirType - Canvas Renderer Module
 * Handles drawing on canvas and stroke visualization
 */

import Config from './config.js';

class CanvasRenderer {
    constructor() {
        this.canvas = null;
        this.ctx = null;
        this.isDrawing = false;
        this.currentStroke = [];
        this.allStrokes = [];
        this.lastPoint = null;
        this.strokeColor = Config.CANVAS.STROKE_COLOR;
        this.strokeWidth = Config.CANVAS.STROKE_WIDTH;
        this.onStrokeCompleteCallback = null;
    }
    
    /**
     * Initialize canvas renderer
     */
    init(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        
        // Set canvas size
        this.resize();
        
        // Setup event listeners
        this.setupEventListeners();
    }
    
    /**
     * Resize canvas to container
     */
    resize() {
        const container = this.canvas.parentElement;
        const rect = container.getBoundingClientRect();
        
        // Set actual canvas size
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        
        // Redraw all strokes after resize
        this.redraw();
    }
    
    /**
     * Setup mouse/touch event listeners
     */
    setupEventListeners() {
        // Mouse events
        this.canvas.addEventListener('mousedown', (e) => this.handlePointerStart(e));
        this.canvas.addEventListener('mousemove', (e) => this.handlePointerMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.handlePointerEnd(e));
        this.canvas.addEventListener('mouseleave', (e) => this.handlePointerEnd(e));
        
        // Touch events
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.handlePointerStart(e.touches[0]);
        });
        
        this.canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.handlePointerMove(e.touches[0]);
        });
        
        this.canvas.addEventListener('touchend', (e) => {
            e.preventDefault();
            this.handlePointerEnd(e);
        });
        
        // Resize handler
        window.addEventListener('resize', () => this.resize());
    }
    
    /**
     * Get point coordinates from event
     */
    getPointFromEvent(event) {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: (event.clientX - rect.left) / rect.width,
            y: (event.clientY - rect.top) / rect.height,
            timestamp: Date.now()
        };
    }
    
    /**
     * Handle pointer start (mouse down / touch start)
     */
    handlePointerStart(event) {
        this.isDrawing = true;
        this.currentStroke = [];
        this.lastPoint = null;
        
        const point = this.getPointFromEvent(event);
        this.addPoint(point);
    }
    
    /**
     * Handle pointer move (mouse move / touch move)
     */
    handlePointerMove(event) {
        if (!this.isDrawing) return;
        
        const point = this.getPointFromEvent(event);
        this.addPoint(point);
    }
    
    /**
     * Handle pointer end (mouse up / touch end)
     */
    handlePointerEnd(event) {
        if (!this.isDrawing) return;
        
        this.isDrawing = false;
        
        if (this.currentStroke.length >= Config.FEATURES.MIN_POINTS) {
            this.allStrokes.push([...this.currentStroke]);
            
            if (this.onStrokeCompleteCallback) {
                this.onStrokeCompleteCallback(this.currentStroke);
            }
        }
        
        this.currentStroke = [];
        this.lastPoint = null;
    }
    
    /**
     * Add point to current stroke
     */
    addPoint(point) {
        this.currentStroke.push(point);
        
        if (this.lastPoint) {
            this.drawSegment(this.lastPoint, point);
        }
        
        this.lastPoint = point;
    }
    
    /**
     * Add point from video stream (hand tracking)
     */
    addVideoPoint(normalizedX, normalizedY) {
        const point = {
            x: normalizedX,
            y: normalizedY,
            timestamp: Date.now()
        };
        
        this.addPoint(point);
    }
    
    /**
     * Start stroke from video
     */
    startVideoStroke() {
        this.isDrawing = true;
        this.currentStroke = [];
        this.lastPoint = null;
    }
    
    /**
     * End stroke from video
     */
    endVideoStroke() {
        if (!this.isDrawing) return null;
        
        this.isDrawing = false;
        
        const stroke = [...this.currentStroke];
        
        if (stroke.length >= Config.FEATURES.MIN_POINTS) {
            this.allStrokes.push(stroke);
            
            if (this.onStrokeCompleteCallback) {
                this.onStrokeCompleteCallback(stroke);
            }
        }
        
        this.currentStroke = [];
        this.lastPoint = null;
        
        return stroke.length >= Config.FEATURES.MIN_POINTS ? stroke : null;
    }
    
    /**
     * Draw line segment between two points
     */
    drawSegment(from, to) {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        ctx.beginPath();
        ctx.strokeStyle = this.strokeColor;
        ctx.lineWidth = this.strokeWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        ctx.moveTo(from.x * width, from.y * height);
        ctx.lineTo(to.x * width, to.y * height);
        ctx.stroke();
    }
    
    /**
     * Draw a complete stroke
     */
    drawStroke(stroke, color = null, width = null) {
        if (stroke.length < 2) return;
        
        const ctx = this.ctx;
        const canvasWidth = this.canvas.width;
        const canvasHeight = this.canvas.height;
        
        ctx.beginPath();
        ctx.strokeStyle = color || this.strokeColor;
        ctx.lineWidth = width || this.strokeWidth;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        ctx.moveTo(stroke[0].x * canvasWidth, stroke[0].y * canvasHeight);
        
        for (let i = 1; i < stroke.length; i++) {
            ctx.lineTo(stroke[i].x * canvasWidth, stroke[i].y * canvasHeight);
        }
        
        ctx.stroke();
    }
    
    /**
     * Redraw all strokes
     */
    redraw() {
        this.clear();
        
        for (const stroke of this.allStrokes) {
            this.drawStroke(stroke);
        }
        
        // Redraw current stroke if drawing
        if (this.isDrawing && this.currentStroke.length > 1) {
            this.drawStroke(this.currentStroke);
        }
    }
    
    /**
     * Clear canvas
     */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    /**
     * Clear all strokes and canvas
     */
    clearAll() {
        this.allStrokes = [];
        this.currentStroke = [];
        this.lastPoint = null;
        this.isDrawing = false;
        this.clear();
    }
    
    /**
     * Undo last stroke
     */
    undo() {
        if (this.allStrokes.length > 0) {
            this.allStrokes.pop();
            this.redraw();
            return true;
        }
        return false;
    }
    
    /**
     * Get all strokes
     */
    getStrokes() {
        return [...this.allStrokes];
    }
    
    /**
     * Get last completed stroke
     */
    getLastStroke() {
        return this.allStrokes.length > 0 ? this.allStrokes[this.allStrokes.length - 1] : null;
    }
    
    /**
     * Get current stroke being drawn
     */
    getCurrentStroke() {
        return this.isDrawing ? [...this.currentStroke] : null;
    }
    
    /**
     * Set stroke color
     */
    setStrokeColor(color) {
        this.strokeColor = color;
    }
    
    /**
     * Set stroke width
     */
    setStrokeWidth(width) {
        this.strokeWidth = width;
    }
    
    /**
     * Set callback for stroke completion
     */
    onStrokeComplete(callback) {
        this.onStrokeCompleteCallback = callback;
    }
    
    /**
     * Get canvas data as base64 image
     */
    toDataURL(type = 'image/png') {
        return this.canvas.toDataURL(type);
    }
    
    /**
     * Export strokes as JSON
     */
    exportStrokes() {
        return JSON.stringify(this.allStrokes);
    }
    
    /**
     * Import strokes from JSON
     */
    importStrokes(json) {
        try {
            this.allStrokes = JSON.parse(json);
            this.redraw();
            return true;
        } catch (error) {
            console.error('Failed to import strokes:', error);
            return false;
        }
    }
    
    /**
     * Check if currently drawing
     */
    isCurrentlyDrawing() {
        return this.isDrawing;
    }
    
    /**
     * Extract features from stroke for prediction
     */
    extractStrokeFeatures(stroke) {
        if (stroke.length < 2) return null;
        
        const features = [];
        
        // Normalize coordinates
        let minX = Infinity, maxX = -Infinity;
        let minY = Infinity, maxY = -Infinity;
        
        for (const point of stroke) {
            minX = Math.min(minX, point.x);
            maxX = Math.max(maxX, point.x);
            minY = Math.min(minY, point.y);
            maxY = Math.max(maxY, point.y);
        }
        
        const rangeX = maxX - minX || 1;
        const rangeY = maxY - minY || 1;
        
        // Resample to fixed number of points
        const numSamples = 50;
        const resampled = this.resampleStroke(stroke, numSamples);
        
        // Calculate features for each point
        for (let i = 0; i < resampled.length; i++) {
            const point = resampled[i];
            
            // Normalized position
            const normalizedX = (point.x - minX) / rangeX;
            const normalizedY = (point.y - minY) / rangeY;
            
            // Velocity
            let velocityX = 0, velocityY = 0;
            if (i > 0) {
                const prev = resampled[i - 1];
                const dt = (point.timestamp - prev.timestamp) / 1000 || 0.001;
                velocityX = (point.x - prev.x) / dt;
                velocityY = (point.y - prev.y) / dt;
            }
            
            // Acceleration
            let accelerationX = 0, accelerationY = 0;
            if (i > 1) {
                const prev = resampled[i - 1];
                const prevPrev = resampled[i - 2];
                const dt1 = (point.timestamp - prev.timestamp) / 1000 || 0.001;
                const dt2 = (prev.timestamp - prevPrev.timestamp) / 1000 || 0.001;
                const prevVelocityX = (prev.x - prevPrev.x) / dt2;
                const prevVelocityY = (prev.y - prevPrev.y) / dt2;
                accelerationX = (velocityX - prevVelocityX) / dt1;
                accelerationY = (velocityY - prevVelocityY) / dt1;
            }
            
            // Curvature
            let curvature = 0;
            if (i > 0 && i < resampled.length - 1) {
                const prev = resampled[i - 1];
                const next = resampled[i + 1];
                
                const v1x = point.x - prev.x;
                const v1y = point.y - prev.y;
                const v2x = next.x - point.x;
                const v2y = next.y - point.y;
                
                const cross = v1x * v2y - v1y * v2x;
                const dot = v1x * v2x + v1y * v2y;
                
                curvature = Math.atan2(cross, dot);
            }
            
            features.push([
                normalizedX,
                normalizedY,
                velocityX,
                velocityY,
                accelerationX,
                accelerationY,
                curvature
            ]);
        }
        
        return features;
    }
    
    /**
     * Resample stroke to fixed number of points
     */
    resampleStroke(stroke, numSamples) {
        if (stroke.length < 2) return stroke;
        
        // Calculate total path length
        let totalLength = 0;
        for (let i = 1; i < stroke.length; i++) {
            const dx = stroke[i].x - stroke[i - 1].x;
            const dy = stroke[i].y - stroke[i - 1].y;
            totalLength += Math.sqrt(dx * dx + dy * dy);
        }
        
        const interval = totalLength / (numSamples - 1);
        const resampled = [stroke[0]];
        
        let accumulatedLength = 0;
        let targetLength = interval;
        
        for (let i = 1; i < stroke.length && resampled.length < numSamples; i++) {
            const prev = stroke[i - 1];
            const curr = stroke[i];
            
            const dx = curr.x - prev.x;
            const dy = curr.y - prev.y;
            const segmentLength = Math.sqrt(dx * dx + dy * dy);
            
            while (accumulatedLength + segmentLength >= targetLength && resampled.length < numSamples) {
                const t = (targetLength - accumulatedLength) / segmentLength;
                
                const newPoint = {
                    x: prev.x + t * dx,
                    y: prev.y + t * dy,
                    timestamp: prev.timestamp + t * (curr.timestamp - prev.timestamp)
                };
                
                resampled.push(newPoint);
                targetLength += interval;
            }
            
            accumulatedLength += segmentLength;
        }
        
        // Ensure we have exactly numSamples points
        while (resampled.length < numSamples) {
            resampled.push(stroke[stroke.length - 1]);
        }
        
        return resampled.slice(0, numSamples);
    }
}

// Export singleton instance
const canvasRenderer = new CanvasRenderer();
export default canvasRenderer;
