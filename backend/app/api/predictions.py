"""
Predictions API Blueprint

Handles prediction-related endpoints for retrieving and analyzing predictions.
"""

from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from marshmallow import Schema, fields, validate, ValidationError
from datetime import datetime, timedelta
from sqlalchemy import func
import uuid
import numpy as np

from app.models import db, Prediction, StrokeSession
from app import limiter
from app.services.predictor import Predictor

predictions_bp = Blueprint("predictions", __name__)


class PredictSchema(Schema):
    """Schema for validating prediction request."""
    features = fields.List(
        fields.List(fields.Float()),
        required=True,
        validate=validate.Length(min=2)
    )
    top_k = fields.Integer(load_default=5, validate=validate.Range(min=1, max=10))


predict_schema = PredictSchema()


@predictions_bp.route("/predict", methods=["POST"])
@limiter.limit("120 per minute")
def predict_character():
    """
    Public endpoint to predict a character from stroke features.
    
    Expects a JSON body with:
        features: 2D array of features (seq_len x 7)
            Each inner array: [normalizedX, normalizedY, velocityX, velocityY, accelerationX, accelerationY, curvature]
        top_k: Number of top predictions to return (default: 5)
    
    Returns:
        JSON response with predicted character, confidence, and alternatives
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate input
        validated = predict_schema.load(data)
        
        features = np.array(validated["features"], dtype=np.float32)
        top_k = validated["top_k"]
        
        # Get predictor instance
        predictor = Predictor(
            redis_url=current_app.config.get("REDIS_URL")
        )
        
        # Make prediction
        result = predictor.predict(features, top_k=top_k)
        
        return jsonify({
            "prediction": result.get("predicted_char", "?"),
            "confidence": result.get("confidence", 0.0),
            "alternatives": result.get("alternatives", []),
            "inference_time": result.get("inference_time_ms", 0),
            "from_cache": result.get("from_cache", False),
            "model_version": result.get("model_version", "unknown")
        })
        
    except ValidationError as e:
        return jsonify({"error": "Validation error", "details": e.messages}), 400
    except Exception as e:
        current_app.logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


class PredictionFilterSchema(Schema):
    """Schema for validating prediction filter parameters."""
    page = fields.Integer(load_default=1, validate=validate.Range(min=1))
    per_page = fields.Integer(load_default=20, validate=validate.Range(min=1, max=100))
    start_date = fields.DateTime(load_default=None)
    end_date = fields.DateTime(load_default=None)
    character = fields.String(load_default=None, validate=validate.Length(max=1))
    min_confidence = fields.Float(load_default=None, validate=validate.Range(min=0, max=1))
    max_confidence = fields.Float(load_default=None, validate=validate.Range(min=0, max=1))


prediction_filter_schema = PredictionFilterSchema()


@predictions_bp.route("/", methods=["GET"])
@jwt_required()
@limiter.limit("60 per minute")
def list_predictions():
    """
    Get paginated list of predictions for the current user.
    
    Query Parameters:
        page: Page number (default: 1)
        per_page: Items per page (default: 20, max: 100)
        start_date: Filter by start date (ISO format)
        end_date: Filter by end date (ISO format)
        character: Filter by predicted character
        min_confidence: Filter by minimum confidence
        max_confidence: Filter by maximum confidence
    
    Returns:
        JSON response with paginated predictions
    """
    try:
        filters = prediction_filter_schema.load(request.args)
        user_id = get_jwt_identity()
        
        # Build query with join to filter by user
        query = Prediction.query.join(
            StrokeSession,
            Prediction.session_id == StrokeSession.id
        ).filter(
            StrokeSession.user_id == uuid.UUID(user_id)
        )
        
        # Apply filters
        if filters.get("start_date"):
            query = query.filter(Prediction.created_at >= filters["start_date"])
        if filters.get("end_date"):
            query = query.filter(Prediction.created_at <= filters["end_date"])
        if filters.get("character"):
            query = query.filter(Prediction.predicted_char == filters["character"])
        if filters.get("min_confidence"):
            query = query.filter(Prediction.confidence >= filters["min_confidence"])
        if filters.get("max_confidence"):
            query = query.filter(Prediction.confidence <= filters["max_confidence"])
        
        # Order by most recent first
        query = query.order_by(Prediction.created_at.desc())
        
        # Paginate
        pagination = query.paginate(
            page=filters["page"],
            per_page=filters["per_page"],
            error_out=False,
        )
        
        return jsonify({
            "predictions": [p.to_dict() for p in pagination.items],
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
        current_app.logger.error(f"Failed to list predictions: {e}")
        return jsonify({"error": "Failed to list predictions"}), 500


@predictions_bp.route("/<uuid:prediction_id>", methods=["GET"])
@jwt_required()
def get_prediction(prediction_id: uuid.UUID):
    """
    Get details of a specific prediction.
    
    Args:
        prediction_id: UUID of the prediction
    
    Returns:
        JSON response with prediction details
    """
    try:
        user_id = get_jwt_identity()
        
        prediction = Prediction.query.join(
            StrokeSession,
            Prediction.session_id == StrokeSession.id
        ).filter(
            Prediction.id == prediction_id,
            StrokeSession.user_id == uuid.UUID(user_id)
        ).first()
        
        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404
        
        return jsonify(prediction.to_dict())
        
    except Exception as e:
        current_app.logger.error(f"Failed to get prediction: {e}")
        return jsonify({"error": "Failed to get prediction"}), 500


@predictions_bp.route("/stats", methods=["GET"])
@jwt_required()
@limiter.limit("30 per minute")
def get_prediction_stats():
    """
    Get prediction statistics for the current user.
    
    Returns:
        JSON response with prediction statistics
    """
    try:
        user_id = get_jwt_identity()
        
        # Base query
        base_query = Prediction.query.join(
            StrokeSession,
            Prediction.session_id == StrokeSession.id
        ).filter(
            StrokeSession.user_id == uuid.UUID(user_id)
        )
        
        # Total predictions
        total_predictions = base_query.count()
        
        # Average confidence
        avg_confidence = db.session.query(
            func.avg(Prediction.confidence)
        ).join(
            StrokeSession,
            Prediction.session_id == StrokeSession.id
        ).filter(
            StrokeSession.user_id == uuid.UUID(user_id)
        ).scalar() or 0
        
        # Average inference time
        avg_inference_time = db.session.query(
            func.avg(Prediction.inference_time_ms)
        ).join(
            StrokeSession,
            Prediction.session_id == StrokeSession.id
        ).filter(
            StrokeSession.user_id == uuid.UUID(user_id)
        ).scalar() or 0
        
        # Character distribution
        char_distribution = db.session.query(
            Prediction.predicted_char,
            func.count(Prediction.id).label("count")
        ).join(
            StrokeSession,
            Prediction.session_id == StrokeSession.id
        ).filter(
            StrokeSession.user_id == uuid.UUID(user_id)
        ).group_by(
            Prediction.predicted_char
        ).order_by(
            func.count(Prediction.id).desc()
        ).limit(20).all()
        
        # Predictions today
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        predictions_today = base_query.filter(
            Prediction.created_at >= today_start
        ).count()
        
        # Predictions this week
        week_start = today_start - timedelta(days=today_start.weekday())
        predictions_week = base_query.filter(
            Prediction.created_at >= week_start
        ).count()
        
        # Confidence distribution
        confidence_buckets = db.session.query(
            func.floor(Prediction.confidence * 10).label("bucket"),
            func.count(Prediction.id).label("count")
        ).join(
            StrokeSession,
            Prediction.session_id == StrokeSession.id
        ).filter(
            StrokeSession.user_id == uuid.UUID(user_id)
        ).group_by(
            "bucket"
        ).all()
        
        confidence_distribution = {
            f"{int(bucket) / 10:.1f}-{(int(bucket) + 1) / 10:.1f}": count
            for bucket, count in confidence_buckets
        }
        
        return jsonify({
            "total_predictions": total_predictions,
            "average_confidence": round(float(avg_confidence), 4),
            "average_inference_time_ms": round(float(avg_inference_time), 2),
            "predictions_today": predictions_today,
            "predictions_this_week": predictions_week,
            "character_distribution": [
                {"character": char, "count": count}
                for char, count in char_distribution
            ],
            "confidence_distribution": confidence_distribution,
        })
        
    except Exception as e:
        current_app.logger.error(f"Failed to get prediction stats: {e}")
        return jsonify({"error": "Failed to get prediction stats"}), 500


@predictions_bp.route("/recent", methods=["GET"])
@jwt_required()
@limiter.limit("120 per minute")
def get_recent_predictions():
    """
    Get recent predictions (last 10) for quick display.
    
    Returns:
        JSON response with recent predictions
    """
    try:
        user_id = get_jwt_identity()
        
        predictions = Prediction.query.join(
            StrokeSession,
            Prediction.session_id == StrokeSession.id
        ).filter(
            StrokeSession.user_id == uuid.UUID(user_id)
        ).order_by(
            Prediction.created_at.desc()
        ).limit(10).all()
        
        return jsonify({
            "predictions": [p.to_dict() for p in predictions],
        })
        
    except Exception as e:
        current_app.logger.error(f"Failed to get recent predictions: {e}")
        return jsonify({"error": "Failed to get recent predictions"}), 500


@predictions_bp.route("/feedback", methods=["POST"])
@jwt_required()
@limiter.limit("30 per minute")
def submit_feedback():
    """
    Submit feedback for a prediction (for model improvement).
    
    Request Body:
        prediction_id: UUID of the prediction
        correct_char: The correct character (if different)
        is_correct: Boolean indicating if prediction was correct
    
    Returns:
        JSON response confirming feedback submission
    """
    try:
        data = request.json
        user_id = get_jwt_identity()
        prediction_id = data.get("prediction_id")
        correct_char = data.get("correct_char")
        is_correct = data.get("is_correct", True)
        
        if not prediction_id:
            return jsonify({"error": "prediction_id is required"}), 400
        
        # Verify prediction belongs to user
        prediction = Prediction.query.join(
            StrokeSession,
            Prediction.session_id == StrokeSession.id
        ).filter(
            Prediction.id == uuid.UUID(prediction_id),
            StrokeSession.user_id == uuid.UUID(user_id)
        ).first()
        
        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404
        
        # Store feedback (in a real system, this would go to a feedback table)
        # For now, we log it for later analysis
        current_app.logger.info(
            "Prediction feedback received",
            extra={
                "prediction_id": prediction_id,
                "user_id": user_id,
                "predicted_char": prediction.predicted_char,
                "correct_char": correct_char,
                "is_correct": is_correct,
            },
        )
        
        return jsonify({
            "message": "Feedback submitted successfully",
            "prediction_id": prediction_id,
        })
        
    except Exception as e:
        current_app.logger.error(f"Failed to submit feedback: {e}")
        return jsonify({"error": "Failed to submit feedback"}), 500
