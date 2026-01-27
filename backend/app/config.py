"""
AirType Configuration Module

This module defines configuration classes for different environments
(development, testing, production) with appropriate settings for each.
"""

import os
from datetime import timedelta
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration class with common settings."""
    
    # Flask Core Settings
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-please-change")
    DEBUG: bool = False
    TESTING: bool = False
    
    # Database Configuration
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL",
        "postgresql://airtype_user:airtype_password@localhost:5432/airtype"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False
    SQLALCHEMY_ENGINE_OPTIONS: dict = {
        "pool_size": 20,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 1800,
        "pool_pre_ping": True,
    }
    
    # Redis Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # JWT Configuration
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "jwt-secret-please-change")
    JWT_ACCESS_TOKEN_EXPIRES: timedelta = timedelta(
        seconds=int(os.getenv("JWT_ACCESS_TOKEN_EXPIRES", "3600"))
    )
    JWT_TOKEN_LOCATION: list = ["headers"]
    JWT_HEADER_NAME: str = "Authorization"
    JWT_HEADER_TYPE: str = "Bearer"
    
    # CORS Configuration
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "/app/models/lstm_v1.0.pth")
    MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")
    
    # Performance Settings
    MAX_STROKE_POINTS: int = int(os.getenv("MAX_STROKE_POINTS", "200"))
    INFERENCE_TIMEOUT_MS: int = int(os.getenv("INFERENCE_TIMEOUT_MS", "100"))
    
    # Rate Limiting
    RATELIMIT_STORAGE_URI: str = os.getenv("REDIS_URL", "memory://")
    RATELIMIT_DEFAULT: str = "100 per minute"
    RATELIMIT_HEADERS_ENABLED: bool = True
    
    # Sentry Configuration
    SENTRY_DSN: Optional[str] = os.getenv("SENTRY_DSN")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Feature Engineering
    SEQUENCE_LENGTH: int = 50  # Fixed number of points per stroke
    CONTEXT_WINDOW: int = 3  # Number of previous strokes for context
    DEDUP_SIMILARITY_THRESHOLD: float = 0.95


class DevelopmentConfig(Config):
    """Development environment configuration."""
    
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    
    # Use SQLite for local development if PostgreSQL not available
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
        "DATABASE_URL",
        "postgresql://airtype_user:airtype_password@localhost:5432/airtype"
    )


class TestingConfig(Config):
    """Testing environment configuration."""
    
    TESTING: bool = True
    DEBUG: bool = True
    
    # Use in-memory SQLite for tests
    SQLALCHEMY_DATABASE_URI: str = "sqlite:///:memory:"
    
    # Disable rate limiting in tests
    RATELIMIT_ENABLED: bool = False
    
    # Use local Redis or memory
    REDIS_URL: str = "memory://"
    
    # JWT settings for testing
    JWT_ACCESS_TOKEN_EXPIRES: timedelta = timedelta(hours=24)


class ProductionConfig(Config):
    """Production environment configuration."""
    
    DEBUG: bool = False
    
    # Stricter security settings
    SESSION_COOKIE_SECURE: bool = True
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SAMESITE: str = "Lax"
    
    # Production database - SSL only for remote databases
    SQLALCHEMY_ENGINE_OPTIONS: dict = {
        "pool_size": 20,
        "max_overflow": 10,
        "pool_timeout": 30,
        "pool_recycle": 1800,
        "pool_pre_ping": True,
    }
    
    # Logging
    LOG_LEVEL: str = "INFO"


# Configuration dictionary for easy access
config = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}
