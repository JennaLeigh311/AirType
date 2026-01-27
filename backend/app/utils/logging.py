"""
AirType Logging Configuration

This module provides structured JSON logging for production environments
with request ID correlation and performance monitoring.
"""

import logging
import sys
import os
import uuid
from typing import Optional
from datetime import datetime

from pythonjsonlogger import jsonlogger
from flask import Flask, g, request


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter that adds additional fields to log records.
    """
    
    def add_fields(self, log_record, record, message_dict):
        """Add custom fields to log record."""
        super().add_fields(log_record, record, message_dict)
        
        # Timestamp
        log_record["timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        # Log level
        log_record["level"] = record.levelname
        
        # Logger name
        log_record["logger"] = record.name
        
        # Request ID (if available)
        try:
            log_record["request_id"] = getattr(g, "request_id", None)
        except RuntimeError:
            # Outside of request context
            pass
        
        # Source location
        log_record["source"] = {
            "file": record.filename,
            "line": record.lineno,
            "function": record.funcName,
        }
        
        # Service information
        log_record["service"] = "airtype-api"
        log_record["environment"] = os.getenv("FLASK_ENV", "development")


class RequestIdFilter(logging.Filter):
    """
    Logging filter that adds request ID to all log records.
    """
    
    def filter(self, record):
        try:
            record.request_id = getattr(g, "request_id", "unknown")
        except RuntimeError:
            record.request_id = "no-context"
        return True


def setup_logging(app: Flask) -> None:
    """
    Configure logging for the Flask application.
    
    Sets up:
    - JSON formatted logging for production
    - Console logging for development
    - Request ID correlation
    - Performance logging
    
    Args:
        app: Flask application instance
    """
    log_level = app.config.get("LOG_LEVEL", "INFO")
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Create handler based on environment
    if app.config.get("ENV") == "production" or os.getenv("FLASK_ENV") == "production":
        # JSON logging for production
        handler = logging.StreamHandler(sys.stdout)
        formatter = CustomJsonFormatter(
            "%(timestamp)s %(level)s %(name)s %(message)s"
        )
        handler.setFormatter(formatter)
    else:
        # Human-readable logging for development
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
    
    # Add request ID filter
    handler.addFilter(RequestIdFilter())
    
    root_logger.addHandler(handler)
    
    # Set Flask logger
    app.logger.handlers = root_logger.handlers
    app.logger.setLevel(getattr(logging, log_level))
    
    # Reduce noise from third-party loggers
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)
    
    app.logger.info(f"Logging configured (level: {log_level})")


def get_request_id() -> str:
    """
    Get the current request ID.
    
    Returns:
        Request ID string
    """
    try:
        if hasattr(g, "request_id"):
            return g.request_id
    except RuntimeError:
        pass
    
    return "no-context"


class PerformanceLogger:
    """
    Context manager for logging performance of code blocks.
    """
    
    def __init__(
        self,
        name: str,
        logger: Optional[logging.Logger] = None,
        threshold_ms: int = 100,
    ):
        """
        Initialize performance logger.
        
        Args:
            name: Name of the operation being measured
            logger: Logger instance (uses root logger if None)
            threshold_ms: Log warning if operation exceeds this threshold
        """
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.threshold_ms = threshold_ms
        self.start_time: Optional[float] = None
    
    def __enter__(self):
        """Start timing."""
        import time
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log result."""
        import time
        
        elapsed_ms = (time.time() - self.start_time) * 1000
        
        log_data = {
            "operation": self.name,
            "duration_ms": round(elapsed_ms, 2),
            "request_id": get_request_id(),
        }
        
        if exc_type is not None:
            log_data["error"] = str(exc_val)
            self.logger.error(f"Operation '{self.name}' failed", extra=log_data)
        elif elapsed_ms > self.threshold_ms:
            self.logger.warning(
                f"Slow operation '{self.name}': {elapsed_ms:.2f}ms",
                extra=log_data,
            )
        else:
            self.logger.debug(
                f"Operation '{self.name}' completed in {elapsed_ms:.2f}ms",
                extra=log_data,
            )
        
        return False  # Don't suppress exceptions


class AuditLogger:
    """
    Logger for audit events (user actions, admin operations).
    """
    
    def __init__(self):
        self.logger = logging.getLogger("airtype.audit")
    
    def log_action(
        self,
        action: str,
        user_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[dict] = None,
    ):
        """
        Log an audit event.
        
        Args:
            action: Action performed (e.g., 'login', 'create_session')
            user_id: ID of user performing action
            resource_type: Type of resource affected
            resource_id: ID of resource affected
            details: Additional details
        """
        log_data = {
            "event_type": "audit",
            "action": action,
            "user_id": user_id,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "details": details or {},
            "request_id": get_request_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        
        try:
            log_data["ip_address"] = request.remote_addr
            log_data["user_agent"] = request.headers.get("User-Agent")
        except RuntimeError:
            pass
        
        self.logger.info(f"Audit: {action}", extra=log_data)


# Global audit logger instance
audit_logger = AuditLogger()
