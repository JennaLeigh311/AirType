"""
AirType Flask Application Factory

This module implements the Flask application factory pattern for creating
and configuring the AirType handwriting recognition API server.
"""

import os
import logging
from typing import Optional, Dict, Any

from flask import Flask, jsonify, request, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_jwt_extended import JWTManager
from flask_socketio import SocketIO
from prometheus_client import make_wsgi_app, Counter, Histogram, Gauge
from werkzeug.middleware.dispatcher import DispatcherMiddleware
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

from app.models import db
from app.utils.logging import setup_logging, get_request_id
from app.config import config

# Initialize extensions
jwt = JWTManager()
socketio = SocketIO()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per minute"],
    storage_uri=os.getenv("REDIS_URL", "memory://"),
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
)
ACTIVE_SESSIONS = Gauge(
    "active_stroke_sessions",
    "Number of active stroke sessions",
)
PREDICTION_COUNT = Counter(
    "stroke_predictions_total",
    "Total stroke predictions",
    ["character", "confidence_bucket"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Model inference latency",
)


def create_app(config_name: Optional[str] = None) -> Flask:
    """
    Create and configure the Flask application.
    
    Args:
        config_name: Configuration environment name ('development', 'testing', 'production')
        
    Returns:
        Configured Flask application instance
    """
    if config_name is None:
        config_name = os.getenv("FLASK_ENV", "development")
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    # Initialize Sentry for error tracking (production only)
    if app.config.get("SENTRY_DSN"):
        sentry_sdk.init(
            dsn=app.config["SENTRY_DSN"],
            integrations=[FlaskIntegration()],
            traces_sample_rate=0.1,
            environment=config_name,
        )
    
    # Setup structured logging
    setup_logging(app)
    
    # Initialize extensions
    db.init_app(app)
    jwt.init_app(app)
    CORS(app, origins=app.config.get("CORS_ORIGINS", "*").split(","))
    limiter.init_app(app)
    
    # SocketIO - don't use Redis message queue in development to avoid eventlet issues
    socketio.init_app(
        app,
        cors_allowed_origins="*",
        async_mode="threading",  # Use threading instead of eventlet for simplicity
        # message_queue disabled for local development
    )
    
    # Register blueprints
    from app.api.strokes import strokes_bp
    from app.api.predictions import predictions_bp
    from app.api.users import users_bp
    from app.api.health import health_bp
    from app.api.training import training_bp
    
    app.register_blueprint(strokes_bp, url_prefix="/api/strokes")
    app.register_blueprint(predictions_bp, url_prefix="/api/predictions")
    app.register_blueprint(users_bp, url_prefix="/api/users")
    app.register_blueprint(health_bp, url_prefix="")
    app.register_blueprint(training_bp, url_prefix="/api/training")
    
    # Prometheus metrics endpoint disabled for local development
    # (causes issues with SocketIO and threading mode)
    # app.wsgi_app = DispatcherMiddleware(
    #     app.wsgi_app,
    #     {"/metrics": make_wsgi_app()},
    # )
    
    # Request hooks for metrics and logging
    @app.before_request
    def before_request() -> None:
        """Track request start time and generate request ID."""
        import time
        import uuid
        
        g.start_time = time.time()
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    
    @app.after_request
    def after_request(response):
        """Log request and record metrics."""
        import time
        
        # Calculate latency
        latency = time.time() - getattr(g, "start_time", time.time())
        
        # Record Prometheus metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.endpoint or "unknown",
            status_code=response.status_code,
        ).inc()
        
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.endpoint or "unknown",
        ).observe(latency)
        
        # Add request ID to response
        response.headers["X-Request-ID"] = getattr(g, "request_id", "unknown")
        
        # Structured logging
        app.logger.info(
            "Request completed",
            extra={
                "request_id": getattr(g, "request_id", "unknown"),
                "method": request.method,
                "path": request.path,
                "status_code": response.status_code,
                "latency_ms": round(latency * 1000, 2),
                "remote_addr": request.remote_addr,
            },
        )
        
        return response
    
    # Error handlers
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request errors."""
        return jsonify({
            "error": "Bad Request",
            "message": str(error.description),
            "request_id": getattr(g, "request_id", "unknown"),
        }), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        """Handle 401 Unauthorized errors."""
        return jsonify({
            "error": "Unauthorized",
            "message": "Authentication required",
            "request_id": getattr(g, "request_id", "unknown"),
        }), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        """Handle 403 Forbidden errors."""
        return jsonify({
            "error": "Forbidden",
            "message": "Access denied",
            "request_id": getattr(g, "request_id", "unknown"),
        }), 403
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found errors."""
        return jsonify({
            "error": "Not Found",
            "message": "Resource not found",
            "request_id": getattr(g, "request_id", "unknown"),
        }), 404
    
    @app.errorhandler(429)
    def rate_limit_exceeded(error):
        """Handle 429 Rate Limit Exceeded errors."""
        return jsonify({
            "error": "Rate Limit Exceeded",
            "message": "Too many requests. Please try again later.",
            "request_id": getattr(g, "request_id", "unknown"),
        }), 429
    
    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 Internal Server errors."""
        app.logger.error(
            f"Internal server error: {error}",
            exc_info=True,
            extra={"request_id": getattr(g, "request_id", "unknown")},
        )
        return jsonify({
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "request_id": getattr(g, "request_id", "unknown"),
        }), 500
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    return app


# WebSocket event handlers
@socketio.on("connect")
def handle_connect():
    """Handle WebSocket connection."""
    from flask_socketio import emit
    from flask import request
    
    emit("connected", {"message": "Connected to AirType WebSocket server"})


@socketio.on("disconnect")
def handle_disconnect():
    """Handle WebSocket disconnection."""
    pass


@socketio.on("stroke_point")
def handle_stroke_point(data: Dict[str, Any]):
    """
    Handle incoming stroke point data.
    
    Args:
        data: Dictionary containing x, y, timestamp, session_id
    """
    from flask_socketio import emit
    from app.services.feature_extractor import FeatureExtractor
    
    try:
        # Validate and process point
        x = float(data.get("x", 0))
        y = float(data.get("y", 0))
        timestamp = int(data.get("timestamp", 0))
        session_id = data.get("session_id")
        
        # Emit acknowledgment
        emit("point_received", {
            "session_id": session_id,
            "timestamp": timestamp,
            "status": "received",
        })
    except Exception as e:
        emit("error", {"message": str(e)})


@socketio.on("stroke_complete")
def handle_stroke_complete(data: Dict[str, Any]):
    """
    Handle stroke completion and trigger prediction.
    
    Args:
        data: Dictionary containing session_id
    """
    from flask_socketio import emit
    from app.services.predictor import Predictor
    
    try:
        session_id = data.get("session_id")
        
        # Get predictor instance and make prediction
        predictor = Predictor()
        result = predictor.predict_from_session(session_id)
        
        emit("prediction", {
            "session_id": session_id,
            "predicted_char": result["predicted_char"],
            "confidence": result["confidence"],
            "alternatives": result["alternatives"],
            "inference_time_ms": result["inference_time_ms"],
        })
        
        PREDICTION_COUNT.labels(
            character=result["predicted_char"],
            confidence_bucket=f"{int(result['confidence'] * 10) / 10:.1f}",
        ).inc()
        
    except Exception as e:
        emit("error", {"message": str(e)})
