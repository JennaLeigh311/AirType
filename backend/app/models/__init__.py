"""
AirType SQLAlchemy Database Models

This module defines all database models for the AirType application including
Users, StrokeSessions, StrokePoints, Predictions, and StrokeFeaturesCache.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import (
    Column, String, Float, Integer, BigInteger, Boolean,
    DateTime, ForeignKey, CheckConstraint, Index, event, Text, TypeDecorator
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB as PG_JSONB, BYTEA as PG_BYTEA
from sqlalchemy.orm import relationship, validates, Mapped, mapped_column
from sqlalchemy.sql import func
import sqlalchemy.types as types
import json

# Initialize SQLAlchemy
db = SQLAlchemy()


# UUID type that works with both PostgreSQL and SQLite
class UUID(TypeDecorator):
    """Platform-independent UUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(36), storing as stringified hex values.
    """
    impl = types.CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(types.CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return str(value)
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(value))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if not isinstance(value, uuid.UUID):
            return uuid.UUID(value)
        else:
            return value


# JSON type that works with both PostgreSQL and SQLite
class JSON(TypeDecorator):
    """Platform-independent JSON type.
    Uses PostgreSQL's JSONB type, otherwise uses TEXT, storing as JSON strings.
    """
    impl = types.TEXT
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_JSONB())
        else:
            return dialect.type_descriptor(types.TEXT())

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        else:
            return json.dumps(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if dialect.name == 'postgresql':
            return value
        else:
            return json.loads(value)


# Binary type that works with both PostgreSQL and SQLite
class Binary(TypeDecorator):
    """Platform-independent binary type.
    Uses PostgreSQL's BYTEA type, otherwise uses BLOB.
    """
    impl = types.BLOB
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_BYTEA())
        else:
            return dialect.type_descriptor(types.BLOB())


class User(db.Model):
    """
    User model for authentication and session tracking.
    
    Attributes:
        id: Unique identifier (UUID)
        username: Unique username (max 50 chars)
        email: Unique email address
        password_hash: Bcrypt hashed password
        is_active: Account active status
        created_at: Account creation timestamp
        updated_at: Last update timestamp
    """
    
    __tablename__ = "users"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        nullable=False,
        index=True,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    
    # Relationships
    stroke_sessions: Mapped[List["StrokeSession"]] = relationship(
        "StrokeSession",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    
    @validates("username")
    def validate_username(self, key: str, username: str) -> str:
        """Validate username format and length."""
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        if len(username) > 50:
            raise ValueError("Username must be at most 50 characters")
        if not username.isalnum() and "_" not in username:
            raise ValueError("Username can only contain alphanumeric characters and underscores")
        return username.lower()
    
    @validates("email")
    def validate_email(self, key: str, email: str) -> str:
        """Validate email format."""
        import re
        email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_regex, email):
            raise ValueError("Invalid email format")
        return email.lower()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary representation."""
        return {
            "id": str(self.id),
            "username": self.username,
            "email": self.email,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }
    
    def __repr__(self) -> str:
        return f"<User {self.username}>"


class StrokeSession(db.Model):
    """
    Stroke session model representing a drawing session.
    
    Attributes:
        id: Unique identifier (UUID)
        user_id: Reference to user who created the session
        started_at: Session start timestamp
        completed_at: Session completion timestamp
        status: Session status ('active', 'completed', 'failed')
    """
    
    __tablename__ = "stroke_sessions"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        nullable=False,
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="stroke_sessions",
    )
    stroke_points: Mapped[List["StrokePoint"]] = relationship(
        "StrokePoint",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="dynamic",
        order_by="StrokePoint.sequence_number",
    )
    predictions: Mapped[List["Prediction"]] = relationship(
        "Prediction",
        back_populates="session",
        cascade="all, delete-orphan",
        lazy="dynamic",
    )
    features_cache: Mapped[Optional["StrokeFeaturesCache"]] = relationship(
        "StrokeFeaturesCache",
        back_populates="session",
        uselist=False,
        cascade="all, delete-orphan",
    )
    
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'completed', 'failed')",
            name="check_session_status",
        ),
        Index("idx_user_started", "user_id", "started_at"),
    )
    
    @validates("status")
    def validate_status(self, key: str, status: str) -> str:
        """Validate session status."""
        valid_statuses = {"active", "completed", "failed"}
        if status not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        return status
    
    def to_dict(self, include_points: bool = False) -> Dict[str, Any]:
        """Convert session to dictionary representation."""
        result = {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "point_count": self.stroke_points.count(),
        }
        if include_points:
            result["points"] = [p.to_dict() for p in self.stroke_points.all()]
        return result
    
    def __repr__(self) -> str:
        return f"<StrokeSession {self.id} status={self.status}>"


class StrokePoint(db.Model):
    """
    Stroke point model representing individual points in a stroke.
    
    Attributes:
        id: Unique identifier (auto-increment)
        session_id: Reference to parent session
        sequence_number: Order of point in sequence
        x: Normalized x coordinate (0-1)
        y: Normalized y coordinate (0-1)
        timestamp_ms: Timestamp in milliseconds
        velocity_x: Horizontal velocity component
        velocity_y: Vertical velocity component
        acceleration: Acceleration magnitude
        curvature: Path curvature at this point
    """
    
    __tablename__ = "stroke_points"
    
    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(),
        ForeignKey("stroke_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    x: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    y: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    timestamp_ms: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    velocity_x: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    velocity_y: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    acceleration: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    curvature: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )
    pen_state: Mapped[int] = mapped_column(
        Integer,
        default=1,  # 1 = pen down, 0 = pen up
        nullable=False,
    )
    
    # Relationships
    session: Mapped["StrokeSession"] = relationship(
        "StrokeSession",
        back_populates="stroke_points",
    )
    
    __table_args__ = (
        CheckConstraint("x >= 0 AND x <= 1", name="check_x_bounds"),
        CheckConstraint("y >= 0 AND y <= 1", name="check_y_bounds"),
        Index("idx_session_sequence", "session_id", "sequence_number"),
    )
    
    @validates("x")
    def validate_x(self, key: str, x: float) -> float:
        """Validate x coordinate is in valid range."""
        if x < 0 or x > 1:
            raise ValueError("x coordinate must be between 0 and 1")
        return x
    
    @validates("y")
    def validate_y(self, key: str, y: float) -> float:
        """Validate y coordinate is in valid range."""
        if y < 0 or y > 1:
            raise ValueError("y coordinate must be between 0 and 1")
        return y
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert point to dictionary representation."""
        return {
            "id": self.id,
            "sequence_number": self.sequence_number,
            "x": self.x,
            "y": self.y,
            "timestamp_ms": self.timestamp_ms,
            "velocity_x": self.velocity_x,
            "velocity_y": self.velocity_y,
            "acceleration": self.acceleration,
            "curvature": self.curvature,
            "pen_state": self.pen_state,
        }
    
    def __repr__(self) -> str:
        return f"<StrokePoint seq={self.sequence_number} ({self.x:.3f}, {self.y:.3f})>"


class Prediction(db.Model):
    """
    Prediction model storing character predictions for stroke sessions.
    
    Attributes:
        id: Unique identifier (UUID)
        session_id: Reference to stroke session
        predicted_char: The predicted character
        confidence: Confidence score (0-1)
        alternatives: JSON array of alternative predictions
        inference_time_ms: Model inference time in milliseconds
        model_version: Version of the model used
        created_at: Prediction timestamp
    """
    
    __tablename__ = "predictions"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(),
        ForeignKey("stroke_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    predicted_char: Mapped[str] = mapped_column(
        String(1),
        nullable=False,
    )
    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    alternatives: Mapped[Optional[Dict]] = mapped_column(
        JSON(),
        nullable=True,
    )
    inference_time_ms: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )
    model_version: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    # Relationships
    session: Mapped["StrokeSession"] = relationship(
        "StrokeSession",
        back_populates="predictions",
    )
    
    __table_args__ = (
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="check_confidence_bounds"),
        Index("idx_confidence", "confidence"),
    )
    
    @validates("confidence")
    def validate_confidence(self, key: str, confidence: float) -> float:
        """Validate confidence is in valid range."""
        if confidence < 0 or confidence > 1:
            raise ValueError("Confidence must be between 0 and 1")
        return confidence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to dictionary representation."""
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "predicted_char": self.predicted_char,
            "confidence": self.confidence,
            "alternatives": self.alternatives,
            "inference_time_ms": self.inference_time_ms,
            "model_version": self.model_version,
            "created_at": self.created_at.isoformat(),
        }
    
    def __repr__(self) -> str:
        return f"<Prediction '{self.predicted_char}' conf={self.confidence:.2f}>"


class StrokeFeaturesCache(db.Model):
    """
    Cache model for preprocessed stroke features.
    
    Used for deduplication and fast feature lookup.
    
    Attributes:
        session_id: Reference to stroke session (primary key)
        normalized_features: Compressed NumPy array of features
        feature_hash: SHA256 hash for deduplication
        created_at: Cache entry creation timestamp
    """
    
    __tablename__ = "stroke_features_cache"
    
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(),
        ForeignKey("stroke_sessions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    normalized_features: Mapped[bytes] = mapped_column(
        Binary(),
        nullable=False,
    )
    feature_hash: Mapped[str] = mapped_column(
        String(64),
        unique=True,
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    # Relationships
    session: Mapped["StrokeSession"] = relationship(
        "StrokeSession",
        back_populates="features_cache",
    )
    
    def __repr__(self) -> str:
        return f"<StrokeFeaturesCache session={self.session_id}>"


# Event listeners for automatic timestamp updates
@event.listens_for(User, "before_update")
def update_user_timestamp(mapper, connection, target):
    """Update the updated_at timestamp before user update."""
    target.updated_at = datetime.utcnow()
