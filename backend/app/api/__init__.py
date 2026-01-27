"""
AirType API Blueprints Package

This package contains all API blueprint modules for the AirType application.
"""

from app.api.strokes import strokes_bp
from app.api.predictions import predictions_bp
from app.api.users import users_bp
from app.api.health import health_bp

__all__ = ["strokes_bp", "predictions_bp", "users_bp", "health_bp"]
