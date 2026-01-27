"""
Users API Blueprint

Handles user authentication, registration, and profile management.
"""

from flask import Blueprint, jsonify, request, current_app
from flask_jwt_extended import (
    jwt_required,
    get_jwt_identity,
    create_access_token,
    create_refresh_token,
)
from marshmallow import Schema, fields, validate, ValidationError
import bcrypt
import uuid

from app.models import db, User
from app import limiter

users_bp = Blueprint("users", __name__)


class RegisterSchema(Schema):
    """Schema for validating user registration."""
    username = fields.String(
        required=True,
        validate=validate.And(
            validate.Length(min=3, max=50),
            validate.Regexp(
                r"^[a-zA-Z0-9_]+$",
                error="Username can only contain letters, numbers, and underscores"
            ),
        ),
    )
    email = fields.Email(required=True)
    password = fields.String(
        required=True,
        validate=validate.Length(min=8, max=128),
    )


class LoginSchema(Schema):
    """Schema for validating user login."""
    username = fields.String(required=True)
    password = fields.String(required=True)


class UpdateProfileSchema(Schema):
    """Schema for validating profile updates."""
    email = fields.Email()
    current_password = fields.String()
    new_password = fields.String(validate=validate.Length(min=8, max=128))


register_schema = RegisterSchema()
login_schema = LoginSchema()
update_profile_schema = UpdateProfileSchema()


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(
        password.encode("utf-8"),
        password_hash.encode("utf-8"),
    )


@users_bp.route("/register", methods=["POST"])
@limiter.limit("5 per minute")
def register():
    """
    Register a new user.
    
    Request Body:
        username: Unique username (3-50 chars, alphanumeric + underscore)
        email: Valid email address
        password: Password (min 8 chars)
    
    Returns:
        JSON response with user details and tokens
    
    Raises:
        400: If validation fails or user already exists
        429: If rate limit exceeded
    """
    try:
        data = register_schema.load(request.json)
        
        # Check if username already exists
        if User.query.filter_by(username=data["username"].lower()).first():
            return jsonify({"error": "Username already taken"}), 400
        
        # Check if email already exists
        if User.query.filter_by(email=data["email"].lower()).first():
            return jsonify({"error": "Email already registered"}), 400
        
        # Create new user
        user = User(
            username=data["username"],
            email=data["email"],
            password_hash=hash_password(data["password"]),
        )
        db.session.add(user)
        db.session.commit()
        
        # Generate tokens
        access_token = create_access_token(identity=str(user.id))
        refresh_token = create_refresh_token(identity=str(user.id))
        
        current_app.logger.info(
            f"New user registered",
            extra={"user_id": str(user.id), "username": user.username},
        )
        
        return jsonify({
            "message": "Registration successful",
            "user": user.to_dict(),
            "access_token": access_token,
            "refresh_token": refresh_token,
        }), 201
        
    except ValidationError as e:
        return jsonify({"error": "Validation error", "details": e.messages}), 400
    except Exception as e:
        current_app.logger.error(f"Registration failed: {e}")
        db.session.rollback()
        return jsonify({"error": "Registration failed"}), 500


@users_bp.route("/login", methods=["POST"])
@limiter.limit("10 per minute")
def login():
    """
    Authenticate user and return JWT tokens.
    
    Request Body:
        username: Username or email
        password: User password
    
    Returns:
        JSON response with user details and tokens
    
    Raises:
        400: If validation fails
        401: If credentials are invalid
        429: If rate limit exceeded
    """
    try:
        data = login_schema.load(request.json)
        
        # Find user by username or email
        user = User.query.filter(
            (User.username == data["username"].lower()) |
            (User.email == data["username"].lower())
        ).first()
        
        if not user or not verify_password(data["password"], user.password_hash):
            return jsonify({"error": "Invalid credentials"}), 401
        
        if not user.is_active:
            return jsonify({"error": "Account is deactivated"}), 401
        
        # Generate tokens
        access_token = create_access_token(identity=str(user.id))
        refresh_token = create_refresh_token(identity=str(user.id))
        
        current_app.logger.info(
            f"User logged in",
            extra={"user_id": str(user.id), "username": user.username},
        )
        
        return jsonify({
            "message": "Login successful",
            "user": user.to_dict(),
            "access_token": access_token,
            "refresh_token": refresh_token,
        })
        
    except ValidationError as e:
        return jsonify({"error": "Validation error", "details": e.messages}), 400
    except Exception as e:
        current_app.logger.error(f"Login failed: {e}")
        return jsonify({"error": "Login failed"}), 500


@users_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    """
    Refresh access token using refresh token.
    
    Returns:
        JSON response with new access token
    
    Raises:
        401: If refresh token is invalid
    """
    try:
        user_id = get_jwt_identity()
        access_token = create_access_token(identity=user_id)
        
        return jsonify({
            "access_token": access_token,
        })
        
    except Exception as e:
        current_app.logger.error(f"Token refresh failed: {e}")
        return jsonify({"error": "Token refresh failed"}), 500


@users_bp.route("/profile", methods=["GET"])
@jwt_required()
def get_profile():
    """
    Get current user's profile.
    
    Returns:
        JSON response with user profile
    
    Raises:
        401: If not authenticated
        404: If user not found
    """
    try:
        user_id = get_jwt_identity()
        user = User.query.get(uuid.UUID(user_id))
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        return jsonify(user.to_dict())
        
    except Exception as e:
        current_app.logger.error(f"Failed to get profile: {e}")
        return jsonify({"error": "Failed to get profile"}), 500


@users_bp.route("/profile", methods=["PATCH"])
@jwt_required()
@limiter.limit("10 per minute")
def update_profile():
    """
    Update current user's profile.
    
    Request Body:
        email: New email address (optional)
        current_password: Current password (required for password change)
        new_password: New password (optional)
    
    Returns:
        JSON response with updated user profile
    
    Raises:
        400: If validation fails
        401: If not authenticated or password incorrect
        404: If user not found
    """
    try:
        data = update_profile_schema.load(request.json)
        user_id = get_jwt_identity()
        user = User.query.get(uuid.UUID(user_id))
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Update email if provided
        if data.get("email"):
            # Check if email already exists
            existing = User.query.filter(
                User.email == data["email"].lower(),
                User.id != user.id,
            ).first()
            if existing:
                return jsonify({"error": "Email already in use"}), 400
            user.email = data["email"]
        
        # Update password if provided
        if data.get("new_password"):
            if not data.get("current_password"):
                return jsonify({"error": "Current password required"}), 400
            
            if not verify_password(data["current_password"], user.password_hash):
                return jsonify({"error": "Current password is incorrect"}), 401
            
            user.password_hash = hash_password(data["new_password"])
        
        db.session.commit()
        
        current_app.logger.info(
            f"User profile updated",
            extra={"user_id": str(user.id)},
        )
        
        return jsonify({
            "message": "Profile updated successfully",
            "user": user.to_dict(),
        })
        
    except ValidationError as e:
        return jsonify({"error": "Validation error", "details": e.messages}), 400
    except Exception as e:
        current_app.logger.error(f"Failed to update profile: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to update profile"}), 500


@users_bp.route("/deactivate", methods=["POST"])
@jwt_required()
def deactivate_account():
    """
    Deactivate current user's account.
    
    Request Body:
        password: Current password for confirmation
    
    Returns:
        JSON response confirming deactivation
    
    Raises:
        400: If password not provided
        401: If not authenticated or password incorrect
        404: If user not found
    """
    try:
        data = request.json
        password = data.get("password")
        
        if not password:
            return jsonify({"error": "Password required"}), 400
        
        user_id = get_jwt_identity()
        user = User.query.get(uuid.UUID(user_id))
        
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        if not verify_password(password, user.password_hash):
            return jsonify({"error": "Incorrect password"}), 401
        
        user.is_active = False
        db.session.commit()
        
        current_app.logger.info(
            f"User account deactivated",
            extra={"user_id": str(user.id)},
        )
        
        return jsonify({"message": "Account deactivated successfully"})
        
    except Exception as e:
        current_app.logger.error(f"Failed to deactivate account: {e}")
        db.session.rollback()
        return jsonify({"error": "Failed to deactivate account"}), 500
