"""
Training API Blueprint

Handles feedback collection, training data management, and model fine-tuning endpoints.
Enables continuous learning through user corrections and active learning.
"""

from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity, get_jwt
from marshmallow import Schema, fields, validate, ValidationError, EXCLUDE
from datetime import datetime, timedelta
from sqlalchemy import func, and_, or_
import uuid
import numpy as np

from app.models import db, TrainingSample, TrainingSession, User
from app import limiter

training_bp = Blueprint("training", __name__)


# ==================
# Validation Schemas
# ==================

class FeedbackSchema(Schema):
    """Schema for validating prediction feedback/correction."""
    
    class Meta:
        unknown = EXCLUDE
    
    stroke_features = fields.List(
        fields.List(fields.Float()),
        required=True,
        validate=validate.Length(min=2)
    )
    predicted_char = fields.String(
        required=True,
        validate=validate.Regexp(r'^[a-zA-Z0-9]$')
    )
    predicted_confidence = fields.Float(
        required=True,
        validate=validate.Range(min=0, max=1)
    )
    alternatives = fields.List(
        fields.Dict(),
        load_default=[]
    )
    actual_char = fields.String(
        required=True,
        validate=validate.Regexp(r'^[a-zA-Z0-9]$')
    )
    correction_type = fields.String(
        load_default="manual",
        validate=validate.OneOf(["manual", "confirmed", "corrected"])
    )
    model_version = fields.String(load_default="1.0.0")
    inference_time_ms = fields.Integer(load_default=0)
    session_id = fields.UUID(load_default=None)
    stroke_metadata = fields.Dict(load_default=None)


class TrainingDataFilterSchema(Schema):
    """Schema for filtering training data queries."""
    
    class Meta:
        unknown = EXCLUDE
    
    page = fields.Integer(load_default=1, validate=validate.Range(min=1))
    per_page = fields.Integer(load_default=50, validate=validate.Range(min=1, max=500))
    char = fields.String(validate=validate.Regexp(r'^[a-zA-Z0-9]$'))
    misclassified_only = fields.Boolean(load_default=False)
    min_confidence = fields.Float(validate=validate.Range(min=0, max=1))
    max_confidence = fields.Float(validate=validate.Range(min=0, max=1))
    correction_type = fields.String(validate=validate.OneOf(["manual", "confirmed", "corrected"]))
    start_date = fields.DateTime()
    end_date = fields.DateTime()


class TrainingRequestSchema(Schema):
    """Schema for requesting model training."""
    
    class Meta:
        unknown = EXCLUDE
    
    min_samples = fields.Integer(load_default=100, validate=validate.Range(min=1))
    char_filter = fields.String(validate=validate.Regexp(r'^[a-zA-Z0-9]$'))
    use_misclassified_only = fields.Boolean(load_default=False)


feedback_schema = FeedbackSchema()
training_data_filter_schema = TrainingDataFilterSchema()
training_request_schema = TrainingRequestSchema()


# ==================
# Endpoints
# ==================

@training_bp.route("/feedback", methods=["POST"])
@limiter.limit("200 per minute")
def submit_feedback():
    """
    Submit prediction feedback/correction for training.
    
    Public endpoint (no auth required) to allow anonymous training data collection.
    Authenticated users get their samples linked to their account.
    
    Request Body:
        stroke_features: 2D array of stroke features
        predicted_char: Character predicted by model
        predicted_confidence: Confidence score
        alternatives: List of alternative predictions
        actual_char: Correct character (user correction)
        correction_type: Type of correction (manual/confirmed/corrected)
        model_version: Version of model that made prediction
        inference_time_ms: Inference time in milliseconds
        session_id: Optional training session ID
        stroke_metadata: Optional raw stroke data
    
    Returns:
        JSON response with created training sample
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate input
        validated = feedback_schema.load(data)
        
        # Get user ID if authenticated (optional)
        user_id = None
        try:
            from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
        except:
            pass  # Anonymous feedback is allowed
        
        # Create training sample
        sample = TrainingSample(
            user_id=uuid.UUID(user_id) if user_id else None,
            session_id=validated.get("session_id"),
            stroke_features=validated["stroke_features"],
            stroke_metadata=validated.get("stroke_metadata"),
            predicted_char=validated["predicted_char"],
            predicted_confidence=validated["predicted_confidence"],
            alternatives=validated.get("alternatives"),
            actual_char=validated["actual_char"],
            correction_type=validated.get("correction_type", "manual"),
            model_version=validated.get("model_version"),
            inference_time_ms=validated.get("inference_time_ms"),
        )
        
        db.session.add(sample)
        db.session.commit()
        
        current_app.logger.info(
            f"Training feedback submitted",
            extra={
                "sample_id": str(sample.id),
                "predicted": sample.predicted_char,
                "actual": sample.actual_char,
                "is_correct": sample.is_correct,
                "user_id": str(user_id) if user_id else "anonymous",
            }
        )
        
        return jsonify({
            "message": "Feedback submitted successfully",
            "sample": sample.to_dict(),
        }), 201
        
    except ValidationError as e:
        return jsonify({"error": "Validation error", "details": e.messages}), 400
    except Exception as e:
        current_app.logger.error(f"Feedback submission failed: {e}", exc_info=True)
        return jsonify({"error": f"Feedback submission failed: {str(e)}"}), 500


@training_bp.route("/data", methods=["GET"])
@jwt_required()
@limiter.limit("30 per minute")
def get_training_data():
    """
    Retrieve training data samples with filtering.
    
    Requires authentication. Returns paginated list of training samples.
    
    Query Parameters:
        page: Page number (default: 1)
        per_page: Items per page (default: 50, max: 500)
        char: Filter by character
        misclassified_only: Show only misclassified samples
        min_confidence: Minimum confidence threshold
        max_confidence: Maximum confidence threshold
        correction_type: Filter by correction type
        start_date: Filter by start date
        end_date: Filter by end date
    
    Returns:
        JSON response with paginated training samples
    """
    try:
        # Validate query parameters
        validated = training_data_filter_schema.load(request.args)
        
        # Base query
        query = TrainingSample.query
        
        # Apply filters
        if validated.get("char"):
            query = query.filter(TrainingSample.actual_char == validated["char"])
        
        if validated.get("misclassified_only"):
            query = query.filter(TrainingSample.predicted_char != TrainingSample.actual_char)
        
        if validated.get("min_confidence") is not None:
            query = query.filter(TrainingSample.predicted_confidence >= validated["min_confidence"])
        
        if validated.get("max_confidence") is not None:
            query = query.filter(TrainingSample.predicted_confidence <= validated["max_confidence"])
        
        if validated.get("correction_type"):
            query = query.filter(TrainingSample.correction_type == validated["correction_type"])
        
        if validated.get("start_date"):
            query = query.filter(TrainingSample.created_at >= validated["start_date"])
        
        if validated.get("end_date"):
            query = query.filter(TrainingSample.created_at <= validated["end_date"])
        
        # Order by most recent first
        query = query.order_by(TrainingSample.created_at.desc())
        
        # Paginate
        page = validated["page"]
        per_page = validated["per_page"]
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        
        return jsonify({
            "samples": [sample.to_dict() for sample in pagination.items],
            "pagination": {
                "page": pagination.page,
                "per_page": pagination.per_page,
                "total": pagination.total,
                "pages": pagination.pages,
                "has_next": pagination.has_next,
                "has_prev": pagination.has_prev,
            }
        })
        
    except ValidationError as e:
        return jsonify({"error": "Validation error", "details": e.messages}), 400
    except Exception as e:
        current_app.logger.error(f"Failed to retrieve training data: {e}", exc_info=True)
        return jsonify({"error": f"Failed to retrieve training data: {str(e)}"}), 500


@training_bp.route("/stats", methods=["GET"])
@limiter.limit("60 per minute")
def get_training_stats():
    """
    Get training statistics and model performance metrics.
    
    Public endpoint showing aggregate statistics.
    
    Returns:
        JSON response with training statistics including:
        - Total samples count
        - Per-character accuracy
        - Overall accuracy
        - Confidence distribution
        - Recent activity
    """
    try:
        # Total samples
        total_samples = db.session.query(func.count(TrainingSample.id)).scalar()
        
        # Correct predictions count
        correct_predictions = db.session.query(func.count(TrainingSample.id)).filter(
            TrainingSample.predicted_char == TrainingSample.actual_char
        ).scalar()
        
        # Overall accuracy
        overall_accuracy = (correct_predictions / total_samples * 100) if total_samples > 0 else 0
        
        # Per-character statistics
        per_char_stats = db.session.query(
            TrainingSample.actual_char,
            func.count(TrainingSample.id).label("total"),
            func.sum(func.cast(TrainingSample.predicted_char == TrainingSample.actual_char, db.Integer)).label("correct"),
            func.avg(TrainingSample.predicted_confidence).label("avg_confidence")
        ).group_by(TrainingSample.actual_char).order_by(func.count(TrainingSample.id).desc()).all()
        
        char_stats = [
            {
                "char": char,
                "total": total,
                "correct": correct or 0,
                "accuracy": round((correct or 0) / total * 100, 2) if total > 0 else 0,
                "avg_confidence": round(float(avg_conf or 0), 4)
            }
            for char, total, correct, avg_conf in per_char_stats
        ]
        
        # Confidence distribution
        confidence_buckets = {
            "0-20%": 0,
            "20-40%": 0,
            "40-60%": 0,
            "60-80%": 0,
            "80-100%": 0
        }
        
        confidence_dist = db.session.query(
            func.count(TrainingSample.id).label("count"),
            func.floor(TrainingSample.predicted_confidence * 5).label("bucket")
        ).group_by("bucket").all()
        
        for count, bucket in confidence_dist:
            bucket_idx = int(bucket)
            if bucket_idx == 0:
                confidence_buckets["0-20%"] = count
            elif bucket_idx == 1:
                confidence_buckets["20-40%"] = count
            elif bucket_idx == 2:
                confidence_buckets["40-60%"] = count
            elif bucket_idx == 3:
                confidence_buckets["60-80%"] = count
            elif bucket_idx >= 4:
                confidence_buckets["80-100%"] = count
        
        # Recent activity (last 7 days)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        recent_samples = db.session.query(func.count(TrainingSample.id)).filter(
            TrainingSample.created_at >= seven_days_ago
        ).scalar()
        
        # Misclassification rate
        misclassified = db.session.query(func.count(TrainingSample.id)).filter(
            TrainingSample.predicted_char != TrainingSample.actual_char
        ).scalar()
        misclassification_rate = (misclassified / total_samples * 100) if total_samples > 0 else 0
        
        return jsonify({
            "total_samples": total_samples,
            "correct_predictions": correct_predictions,
            "overall_accuracy": round(overall_accuracy, 2),
            "misclassification_rate": round(misclassification_rate, 2),
            "per_character": char_stats,
            "confidence_distribution": confidence_buckets,
            "recent_activity": {
                "last_7_days": recent_samples
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Failed to retrieve training stats: {e}", exc_info=True)
        return jsonify({"error": f"Failed to retrieve training stats: {str(e)}"}), 500


@training_bp.route("/train", methods=["POST"])
@limiter.limit("10 per hour")
def trigger_training():
    """
    Trigger model training/fine-tuning with collected training data.
    
    Supports both full training and incremental fine-tuning with GPU acceleration.
    
    Request Body:
        min_samples: Minimum samples required to start training (default: 50)
        epochs: Number of training epochs (default: 10)
        augment: Whether to apply data augmentation (default: true)
        augmentation_factor: Augmentation multiplier (default: 5)
        fine_tune: Use fine-tuning mode with lower LR (default: false)
        use_misclassified_only: Train only on misclassified samples
    
    Returns:
        JSON response with training results
    """
    try:
        data = request.get_json() or {}
        
        min_samples = data.get("min_samples", 50)
        epochs = min(data.get("epochs", 10), 50)  # Cap at 50 epochs
        augment = data.get("augment", True)
        augmentation_factor = min(data.get("augmentation_factor", 5), 10)
        fine_tune = data.get("fine_tune", False)
        use_misclassified_only = data.get("use_misclassified_only", False)
        
        # Get training samples
        query = TrainingSample.query
        
        if use_misclassified_only:
            query = query.filter(TrainingSample.predicted_char != TrainingSample.actual_char)
        
        samples = query.all()
        sample_count = len(samples)
        
        if sample_count < min_samples:
            return jsonify({
                "error": f"Not enough samples for training. Found {sample_count}, need at least {min_samples}",
                "current_samples": sample_count,
                "required_samples": min_samples
            }), 400
        
        # Prepare data
        features_list = [sample.stroke_features for sample in samples]
        labels_list = [sample.actual_char for sample in samples]
        
        # Import and run training
        from app.ml.fast_training import get_trainer
        trainer = get_trainer()
        
        if fine_tune:
            # Fine-tuning with lower learning rate
            result = trainer.fine_tune(
                features_list,
                labels_list,
                epochs=epochs,
                learning_rate=0.0001
            )
        else:
            # Full training with augmentation
            result = trainer.train(
                features_list,
                labels_list,
                epochs=epochs,
                learning_rate=0.001,
                augment=augment,
                augmentation_factor=augmentation_factor
            )
        
        # Create training session record
        user_id = None
        try:
            from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
        except:
            pass
        
        training_session = TrainingSession(
            user_id=uuid.UUID(user_id) if user_id else None,
            model_version=current_app.config.get("MODEL_VERSION", "1.0.0"),
            samples_count=sample_count,
            status="completed",
        )
        
        db.session.add(training_session)
        db.session.commit()
        
        current_app.logger.info(
            f"Training completed",
            extra={
                "session_id": str(training_session.id),
                "samples": sample_count,
                "accuracy": result.get("final_val_accuracy"),
                "device": result.get("device"),
            }
        )
        
        return jsonify({
            "message": "Training completed successfully",
            "session": training_session.to_dict(),
            "results": result
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Training failed: {e}", exc_info=True)
        return jsonify({"error": f"Training failed: {str(e)}"}), 500


@training_bp.route("/export", methods=["GET"])
@jwt_required()
@limiter.limit("10 per hour")
def export_training_data():
    """
    Export training data in format suitable for model training.
    
    Requires authentication. Returns data in NumPy-compatible format.
    
    Returns:
        JSON response with features (X) and labels (y) arrays
    """
    try:
        # Get all training samples
        samples = TrainingSample.query.order_by(TrainingSample.created_at.asc()).all()
        
        if not samples:
            return jsonify({"error": "No training data available"}), 404
        
        # Prepare data
        features_list = []
        labels_list = []
        
        for sample in samples:
            features_list.append(sample.stroke_features)
            labels_list.append(sample.actual_char)
        
        return jsonify({
            "total_samples": len(samples),
            "features": features_list,
            "labels": labels_list,
            "format": {
                "features": "List of 2D arrays (samples x features)",
                "labels": "List of characters"
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Failed to export training data: {e}", exc_info=True)
        return jsonify({"error": f"Failed to export training data: {str(e)}"}), 500


@training_bp.route("/suggestions", methods=["GET"])
@limiter.limit("60 per minute")
def get_practice_suggestions():
    """
    Get active learning suggestions for which characters to practice.
    
    Analyzes collected data to identify:
    - Underrepresented characters
    - Low-accuracy characters
    - Commonly confused character pairs
    
    Returns:
        JSON response with prioritized character suggestions
    """
    try:
        from app.ml.active_learning import ActiveLearningStrategy
        
        # Get all samples
        samples = TrainingSample.query.all()
        
        if not samples:
            # No data yet - suggest starting with basics
            return jsonify({
                "suggestions": [
                    {"char": c, "score": 10, "reasons": ["No data collected yet"], "current_count": 0}
                    for c in "aeiou12345"
                ],
                "practice_session": list("aeiouAEIOU1234567890"),
                "total_samples": 0,
                "message": "Start by collecting some samples! Draw characters and submit corrections."
            })
        
        # Convert to dicts for analysis
        sample_dicts = [
            {
                "actual_char": s.actual_char,
                "predicted_char": s.predicted_char,
                "predicted_confidence": s.predicted_confidence
            }
            for s in samples
        ]
        
        # Run analysis
        strategy = ActiveLearningStrategy()
        analysis = strategy.analyze_samples(sample_dicts)
        suggestions = strategy.suggest_characters(analysis, num_suggestions=15)
        practice_session = strategy.get_practice_session(analysis, session_length=20)
        
        return jsonify({
            "suggestions": suggestions,
            "practice_session": practice_session,
            "analysis": {
                "total_samples": analysis["total_samples"],
                "unique_chars_covered": analysis["unique_chars_covered"],
                "chars_needing_samples": analysis["chars_below_minimum"],
                "top_confusions": analysis["top_confusions"][:10],
                "low_confidence_count": analysis["low_confidence_count"]
            }
        })
        
    except Exception as e:
        current_app.logger.error(f"Failed to get suggestions: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get suggestions: {str(e)}"}), 500


@training_bp.route("/training-info", methods=["GET"])
@limiter.limit("60 per minute")
def get_training_info():
    """
    Get information about training capabilities and status.
    
    Returns:
        JSON response with device info, GPU availability, and training status
    """
    try:
        from app.ml.fast_training import get_trainer
        
        trainer = get_trainer()
        info = trainer.get_training_info()
        
        # Add sample counts
        total_samples = db.session.query(func.count(TrainingSample.id)).scalar()
        
        return jsonify({
            **info,
            "total_samples": total_samples,
            "ready_for_training": total_samples >= 50,
            "recommended_action": (
                "Collect more samples" if total_samples < 50
                else "Ready to train!" if total_samples < 200
                else "Consider training or fine-tuning"
            )
        })
        
    except Exception as e:
        current_app.logger.error(f"Failed to get training info: {e}", exc_info=True)
        return jsonify({"error": f"Failed to get training info: {str(e)}"}), 500


@training_bp.route("/batch-feedback", methods=["POST"])
@limiter.limit("50 per minute")
def submit_batch_feedback():
    """
    Submit multiple feedback samples in one request.
    
    Efficient for batch data collection sessions.
    
    Request Body:
        samples: List of feedback objects (same format as /feedback endpoint)
    
    Returns:
        JSON response with submission summary
    """
    try:
        data = request.get_json()
        
        if not data or "samples" not in data:
            return jsonify({"error": "No samples provided"}), 400
        
        samples_data = data["samples"]
        
        if not isinstance(samples_data, list):
            return jsonify({"error": "samples must be a list"}), 400
        
        if len(samples_data) > 100:
            return jsonify({"error": "Maximum 100 samples per batch"}), 400
        
        # Get user ID if authenticated
        user_id = None
        try:
            from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
            verify_jwt_in_request(optional=True)
            user_id = get_jwt_identity()
        except:
            pass
        
        # Process each sample
        created = []
        errors = []
        
        for i, sample_data in enumerate(samples_data):
            try:
                # Validate required fields
                if not all(k in sample_data for k in ["stroke_features", "predicted_char", "actual_char"]):
                    errors.append({"index": i, "error": "Missing required fields"})
                    continue
                
                sample = TrainingSample(
                    user_id=uuid.UUID(user_id) if user_id else None,
                    stroke_features=sample_data["stroke_features"],
                    stroke_metadata=sample_data.get("stroke_metadata"),
                    predicted_char=sample_data["predicted_char"],
                    predicted_confidence=sample_data.get("predicted_confidence", 0.5),
                    alternatives=sample_data.get("alternatives"),
                    actual_char=sample_data["actual_char"],
                    correction_type=sample_data.get("correction_type", "manual"),
                    model_version=sample_data.get("model_version", "1.0.0"),
                    inference_time_ms=sample_data.get("inference_time_ms", 0),
                )
                
                db.session.add(sample)
                created.append(str(sample.id))
                
            except Exception as e:
                errors.append({"index": i, "error": str(e)})
        
        db.session.commit()
        
        current_app.logger.info(
            f"Batch feedback submitted",
            extra={
                "created_count": len(created),
                "error_count": len(errors),
                "user_id": str(user_id) if user_id else "anonymous",
            }
        )
        
        return jsonify({
            "message": f"Processed {len(samples_data)} samples",
            "created": len(created),
            "errors": len(errors),
            "error_details": errors if errors else None
        }), 201
        
    except Exception as e:
        current_app.logger.error(f"Batch feedback failed: {e}", exc_info=True)
        return jsonify({"error": f"Batch feedback failed: {str(e)}"}), 500

