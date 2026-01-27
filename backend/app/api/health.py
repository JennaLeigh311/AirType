"""
Health Check API Blueprint

Provides health check endpoints for monitoring and load balancer probes.
"""

from flask import Blueprint, jsonify, current_app
from datetime import datetime
import redis

from app.models import db

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health_check():
    """
    Basic health check endpoint.
    
    Returns:
        JSON response with service status
    """
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "airtype-api",
    })


@health_bp.route("/health/detailed", methods=["GET"])
def detailed_health_check():
    """
    Detailed health check including database and Redis connectivity.
    
    Returns:
        JSON response with detailed service status
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "airtype-api",
        "checks": {},
    }
    
    # Check database connection
    try:
        db.session.execute(db.text("SELECT 1"))
        health_status["checks"]["database"] = {
            "status": "healthy",
            "message": "PostgreSQL connection successful",
        }
    except Exception as e:
        health_status["checks"]["database"] = {
            "status": "unhealthy",
            "message": str(e),
        }
        health_status["status"] = "degraded"
    
    # Check Redis connection
    try:
        redis_url = current_app.config.get("REDIS_URL", "redis://localhost:6379/0")
        if not redis_url.startswith("memory://"):
            r = redis.from_url(redis_url)
            r.ping()
            health_status["checks"]["redis"] = {
                "status": "healthy",
                "message": "Redis connection successful",
            }
        else:
            health_status["checks"]["redis"] = {
                "status": "healthy",
                "message": "Using in-memory storage",
            }
    except Exception as e:
        health_status["checks"]["redis"] = {
            "status": "unhealthy",
            "message": str(e),
        }
        health_status["status"] = "degraded"
    
    # Check model availability
    try:
        from app.services.predictor import Predictor
        predictor = Predictor()
        if predictor.is_model_loaded():
            health_status["checks"]["model"] = {
                "status": "healthy",
                "message": "Model loaded successfully",
                "version": current_app.config.get("MODEL_VERSION", "unknown"),
            }
        else:
            health_status["checks"]["model"] = {
                "status": "degraded",
                "message": "Model not loaded - using fallback",
            }
    except Exception as e:
        health_status["checks"]["model"] = {
            "status": "unhealthy",
            "message": str(e),
        }
    
    status_code = 200 if health_status["status"] == "healthy" else 503
    return jsonify(health_status), status_code


@health_bp.route("/ready", methods=["GET"])
def readiness_check():
    """
    Readiness probe for Kubernetes/container orchestration.
    
    Returns:
        JSON response indicating if service is ready to accept traffic
    """
    try:
        # Verify database is accessible
        db.session.execute(db.text("SELECT 1"))
        
        return jsonify({
            "ready": True,
            "timestamp": datetime.utcnow().isoformat(),
        })
    except Exception as e:
        return jsonify({
            "ready": False,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }), 503


@health_bp.route("/live", methods=["GET"])
def liveness_check():
    """
    Liveness probe for container orchestration.
    
    Returns:
        JSON response indicating if service is alive
    """
    return jsonify({
        "alive": True,
        "timestamp": datetime.utcnow().isoformat(),
    })
