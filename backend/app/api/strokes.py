"""
Strokes API Blueprint

Handles stroke session management, point collection, and stroke completion.
"""

from flask import Blueprint, jsonify, request, current_app, g
from flask_jwt_extended import jwt_required, get_jwt_identity
from marshmallow import Schema, fields, validate, ValidationError
from datetime import datetime
from typing import Dict, Any, List
import uuid

from app.models import db, StrokeSession, StrokePoint, User
from app.services.feature_extractor import FeatureExtractor
from app.services.predictor import Predictor
from app.services.deduplicator import Deduplicator
from app.utils.validation import validate_request
from app import limiter

strokes_bp = Blueprint("strokes", __name__)


# Marshmallow Schemas for validation
class StartSessionSchema(Schema):
    """Schema for validating stroke session start requests."""
    pass  # No required fields for starting a session


class StrokePointSchema(Schema):
    """Schema for validating individual stroke points."""
    x = fields.Float(required=True, validate=validate.Range(min=0, max=1))
    y = fields.Float(required=True, validate=validate.Range(min=0, max=1))
    timestamp = fields.Integer(required=True, validate=validate.Range(min=0))
    pressure_estimate = fields.Float(validate=validate.Range(min=0, max=1))


class AddPointsSchema(Schema):
    """Schema for validating add points requests."""
    session_id = fields.UUID(required=True)
    points = fields.List(fields.Nested(StrokePointSchema), required=True, validate=validate.Length(min=1, max=100))


class CompleteStrokeSchema(Schema):
    """Schema for validating stroke completion requests."""
    session_id = fields.UUID(required=True)


class HistoryFilterSchema(Schema):
    """Schema for validating history filter parameters."""
    page = fields.Integer(load_default=1, validate=validate.Range(min=1))
    per_page = fields.Integer(load_default=20, validate=validate.Range(min=1, max=100))
    start_date = fields.DateTime(load_default=None)
    end_date = fields.DateTime(load_default=None)
    character = fields.String(load_default=None, validate=validate.Length(max=1))
    min_confidence = fields.Float(load_default=None, validate=validate.Range(min=0, max=1))


# Initialize schemas
start_session_schema = StartSessionSchema()
add_points_schema = AddPointsSchema()
complete_stroke_schema = CompleteStrokeSchema()
history_filter_schema = HistoryFilterSchema()


@strokes_bp.route("/start", methods=["POST"])
@jwt_required()
@limiter.limit("20 per minute")
def start_session():
    """
    Initialize a new stroke session.
    
    Returns:
        JSON response with session_id and WebSocket URL
    
    Raises:
        401: If not authenticated
        429: If rate limit exceeded
    """
    try:
        user_id = get_jwt_identity()
        
        # Create new session
        session = StrokeSession(
            user_id=uuid.UUID(user_id),
            status="active",
        )
        db.session.add(session)
        db.session.commit()
        
        # Generate WebSocket URL
        ws_host = request.host.replace("http", "ws").replace("https", "wss")
        ws_url = f"wss://{ws_host}/ws/strokes/{session.id}"
        
        current_app.logger.info(
            f"Started stroke session",
            extra={
                "session_id": str(session.id),
                "user_id": user_id,
                "request_id": getattr(g, "request_id", "unknown"),
            },
        )
        
        return jsonify({
            "session_id": str(session.id),
            "websocket_url": ws_url,
            "status": "active",
            "started_at": session.started_at.isoformat(),
        }), 201
        
    except ValidationError as e:
        return jsonify({"error": "Validation error", "details": e.messages}), 400
    except Exception as e:
        current_app.logger.error(f"Failed to start session: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to start session"}), 500


@strokes_bp.route("/points", methods=["POST"])
@jwt_required()
@limiter.limit("500 per minute")
def add_points():
    """
    Add stroke points to an active session.
    
    Request Body:
        session_id: UUID of the active session
        points: List of point objects with x, y, timestamp
    
    Returns:
        JSON response with acknowledgment and point IDs
    
    Raises:
        400: If validation fails
        401: If not authenticated
        404: If session not found
        429: If rate limit exceeded
    """
    try:
        data = add_points_schema.load(request.json)
        user_id = get_jwt_identity()
        
        # Verify session exists and belongs to user
        session = StrokeSession.query.filter_by(
            id=data["session_id"],
            user_id=uuid.UUID(user_id),
            status="active",
        ).first()
        
        if not session:
            return jsonify({"error": "Session not found or not active"}), 404
        
        # Check max points limit
        current_count = session.stroke_points.count()
        max_points = current_app.config.get("MAX_STROKE_POINTS", 200)
        
        if current_count + len(data["points"]) > max_points:
            return jsonify({
                "error": f"Max points limit ({max_points}) would be exceeded",
                "current_count": current_count,
            }), 400
        
        # Add points to database
        point_ids = []
        feature_extractor = FeatureExtractor()
        
        for i, point_data in enumerate(data["points"]):
            sequence_number = current_count + i + 1
            
            # Calculate kinematic features if we have enough points
            velocity_x, velocity_y, acceleration, curvature = None, None, None, None
            
            if sequence_number >= 3:
                # Get previous points for feature calculation
                prev_points = session.stroke_points.order_by(
                    StrokePoint.sequence_number.desc()
                ).limit(2).all()
                
                if len(prev_points) >= 2:
                    features = feature_extractor.calculate_point_features(
                        prev_points[1].to_dict(),
                        prev_points[0].to_dict(),
                        point_data,
                    )
                    velocity_x = features.get("velocity_x")
                    velocity_y = features.get("velocity_y")
                    acceleration = features.get("acceleration")
                    curvature = features.get("curvature")
            
            point = StrokePoint(
                session_id=session.id,
                sequence_number=sequence_number,
                x=point_data["x"],
                y=point_data["y"],
                timestamp_ms=point_data["timestamp"],
                velocity_x=velocity_x,
                velocity_y=velocity_y,
                acceleration=acceleration,
                curvature=curvature,
            )
            db.session.add(point)
            db.session.flush()  # Get the ID
            point_ids.append(point.id)
        
        db.session.commit()
        
        return jsonify({
            "status": "received",
            "session_id": str(session.id),
            "point_ids": point_ids,
            "total_points": current_count + len(data["points"]),
        }), 201
        
    except ValidationError as e:
        return jsonify({"error": "Validation error", "details": e.messages}), 400
    except Exception as e:
        current_app.logger.error(f"Failed to add points: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to add points"}), 500


@strokes_bp.route("/complete", methods=["POST"])
@jwt_required()
@limiter.limit("30 per minute")
def complete_stroke():
    """
    Complete a stroke session and trigger prediction.
    
    Request Body:
        session_id: UUID of the session to complete
    
    Returns:
        JSON response with prediction results
    
    Raises:
        400: If validation fails or session has no points
        401: If not authenticated
        404: If session not found
        429: If rate limit exceeded
    """
    try:
        data = complete_stroke_schema.load(request.json)
        user_id = get_jwt_identity()
        
        # Verify session exists and belongs to user
        session = StrokeSession.query.filter_by(
            id=data["session_id"],
            user_id=uuid.UUID(user_id),
        ).first()
        
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        if session.status != "active":
            return jsonify({"error": f"Session is already {session.status}"}), 400
        
        # Verify session has points
        point_count = session.stroke_points.count()
        if point_count < 5:
            return jsonify({
                "error": "Session needs at least 5 points for prediction",
                "current_count": point_count,
            }), 400
        
        # Extract features
        feature_extractor = FeatureExtractor()
        points = session.stroke_points.order_by(StrokePoint.sequence_number).all()
        features = feature_extractor.extract_sequence_features([p.to_dict() for p in points])
        
        # Check for duplicates
        deduplicator = Deduplicator()
        duplicate = deduplicator.check_duplicate(features, user_id)
        
        if duplicate:
            # Return cached prediction for duplicate stroke
            current_app.logger.info(
                f"Duplicate stroke detected",
                extra={
                    "session_id": str(session.id),
                    "duplicate_of": str(duplicate["session_id"]),
                },
            )
            session.status = "completed"
            session.completed_at = datetime.utcnow()
            db.session.commit()
            
            return jsonify({
                "session_id": str(session.id),
                "predicted_char": duplicate["predicted_char"],
                "confidence": duplicate["confidence"],
                "alternatives": duplicate["alternatives"],
                "is_duplicate": True,
                "duplicate_of": str(duplicate["session_id"]),
            })
        
        # Make prediction
        predictor = Predictor()
        prediction_result = predictor.predict(features)
        
        # Cache features for future deduplication
        deduplicator.cache_features(session.id, features, prediction_result)
        
        # Store prediction in database
        from app.models import Prediction
        prediction = Prediction(
            session_id=session.id,
            predicted_char=prediction_result["predicted_char"],
            confidence=prediction_result["confidence"],
            alternatives=prediction_result["alternatives"],
            inference_time_ms=prediction_result["inference_time_ms"],
            model_version=current_app.config.get("MODEL_VERSION", "1.0.0"),
        )
        db.session.add(prediction)
        
        # Update session status
        session.status = "completed"
        session.completed_at = datetime.utcnow()
        db.session.commit()
        
        current_app.logger.info(
            f"Stroke prediction completed",
            extra={
                "session_id": str(session.id),
                "predicted_char": prediction_result["predicted_char"],
                "confidence": prediction_result["confidence"],
                "inference_time_ms": prediction_result["inference_time_ms"],
            },
        )
        
        return jsonify({
            "session_id": str(session.id),
            "predicted_char": prediction_result["predicted_char"],
            "confidence": prediction_result["confidence"],
            "alternatives": prediction_result["alternatives"],
            "inference_time_ms": prediction_result["inference_time_ms"],
            "is_duplicate": False,
        })
        
    except ValidationError as e:
        return jsonify({"error": "Validation error", "details": e.messages}), 400
    except Exception as e:
        current_app.logger.error(f"Failed to complete stroke: {e}", exc_info=True)
        db.session.rollback()
        return jsonify({"error": "Failed to complete stroke"}), 500


@strokes_bp.route("/history", methods=["GET"])
@jwt_required()
@limiter.limit("60 per minute")
def get_history():
    """
    Get paginated stroke history with predictions.
    
    Query Parameters:
        page: Page number (default: 1)
        per_page: Items per page (default: 20, max: 100)
        start_date: Filter by start date (ISO format)
        end_date: Filter by end date (ISO format)
        character: Filter by predicted character
        min_confidence: Filter by minimum confidence score
    
    Returns:
        JSON response with paginated stroke history
    
    Raises:
        401: If not authenticated
        429: If rate limit exceeded
    """
    try:
        filters = history_filter_schema.load(request.args)
        user_id = get_jwt_identity()
        
        # Build query
        query = StrokeSession.query.filter_by(
            user_id=uuid.UUID(user_id),
            status="completed",
        )
        
        # Apply date filters
        if filters.get("start_date"):
            query = query.filter(StrokeSession.started_at >= filters["start_date"])
        if filters.get("end_date"):
            query = query.filter(StrokeSession.started_at <= filters["end_date"])
        
        # Apply character and confidence filters via join with predictions
        if filters.get("character") or filters.get("min_confidence"):
            from app.models import Prediction
            query = query.join(Prediction)
            
            if filters.get("character"):
                query = query.filter(Prediction.predicted_char == filters["character"])
            if filters.get("min_confidence"):
                query = query.filter(Prediction.confidence >= filters["min_confidence"])
        
        # Order by most recent first
        query = query.order_by(StrokeSession.started_at.desc())
        
        # Paginate
        pagination = query.paginate(
            page=filters["page"],
            per_page=filters["per_page"],
            error_out=False,
        )
        
        # Build response
        sessions_data = []
        for session in pagination.items:
            session_dict = session.to_dict()
            
            # Include latest prediction
            latest_prediction = session.predictions.order_by(
                db.text("created_at DESC")
            ).first()
            
            if latest_prediction:
                session_dict["prediction"] = latest_prediction.to_dict()
            
            sessions_data.append(session_dict)
        
        return jsonify({
            "sessions": sessions_data,
            "pagination": {
                "page": pagination.page,
                "per_page": pagination.per_page,
                "total_pages": pagination.pages,
                "total_items": pagination.total,
                "has_next": pagination.has_next,
                "has_prev": pagination.has_prev,
            },
        })
        
    except ValidationError as e:
        return jsonify({"error": "Validation error", "details": e.messages}), 400
    except Exception as e:
        current_app.logger.error(f"Failed to get history: {e}")
        return jsonify({"error": "Failed to get history"}), 500


@strokes_bp.route("/<uuid:session_id>", methods=["GET"])
@jwt_required()
def get_session(session_id: uuid.UUID):
    """
    Get details of a specific stroke session.
    
    Args:
        session_id: UUID of the session
    
    Returns:
        JSON response with session details including points
    
    Raises:
        401: If not authenticated
        404: If session not found
    """
    try:
        user_id = get_jwt_identity()
        
        session = StrokeSession.query.filter_by(
            id=session_id,
            user_id=uuid.UUID(user_id),
        ).first()
        
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        return jsonify(session.to_dict(include_points=True))
        
    except Exception as e:
        current_app.logger.error(f"Failed to get session: {e}")
        return jsonify({"error": "Failed to get session"}), 500


@strokes_bp.route("/<uuid:session_id>", methods=["DELETE"])
@jwt_required()
def delete_session(session_id: uuid.UUID):
    """
    Delete a stroke session.
    
    Args:
        session_id: UUID of the session to delete
    
    Returns:
        JSON response confirming deletion
    
    Raises:
        401: If not authenticated
        404: If session not found
    """
    try:
        user_id = get_jwt_identity()
        
        session = StrokeSession.query.filter_by(
            id=session_id,
            user_id=uuid.UUID(user_id),
        ).first()
        
        if not session:
            return jsonify({"error": "Session not found"}), 404
        
        db.session.delete(session)
        db.session.commit()
        
        return jsonify({"message": "Session deleted successfully"}), 200
        
    except Exception as e:
        current_app.logger.error(f"Failed to delete session: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to delete session"}), 500
