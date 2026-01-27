"""
AirType Request Validation

This module provides request validation utilities using Marshmallow schemas.
"""

from typing import Callable, Any, Type
from functools import wraps

from flask import request, jsonify, g
from marshmallow import Schema, ValidationError


def validate_request(schema: Type[Schema], location: str = "json"):
    """
    Decorator for validating request data against a Marshmallow schema.
    
    Args:
        schema: Marshmallow schema class to validate against
        location: Where to get data from ('json', 'args', 'form')
    
    Returns:
        Decorator function
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def decorated_function(*args, **kwargs):
            schema_instance = schema()
            
            # Get data from appropriate location
            if location == "json":
                data = request.json or {}
            elif location == "args":
                data = request.args.to_dict()
            elif location == "form":
                data = request.form.to_dict()
            else:
                data = {}
            
            try:
                # Validate and deserialize
                validated_data = schema_instance.load(data)
                
                # Store validated data in g
                g.validated_data = validated_data
                
            except ValidationError as e:
                return jsonify({
                    "error": "Validation error",
                    "details": e.messages,
                    "request_id": getattr(g, "request_id", "unknown"),
                }), 400
            
            return f(*args, **kwargs)
        
        return decorated_function
    
    return decorator


def validate_uuid(value: str, name: str = "ID") -> bool:
    """
    Validate that a string is a valid UUID.
    
    Args:
        value: String to validate
        name: Name of the field for error messages
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If invalid UUID
    """
    import uuid
    
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, TypeError):
        raise ValueError(f"Invalid {name}: must be a valid UUID")


def validate_coordinate(value: float, name: str = "coordinate") -> bool:
    """
    Validate that a coordinate is in valid range [0, 1].
    
    Args:
        value: Coordinate value
        name: Name of the field for error messages
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If out of range
    """
    if value < 0 or value > 1:
        raise ValueError(f"Invalid {name}: must be between 0 and 1")
    return True


def validate_timestamp(value: int, name: str = "timestamp") -> bool:
    """
    Validate that a timestamp is positive.
    
    Args:
        value: Timestamp value in milliseconds
        name: Name of the field for error messages
    
    Returns:
        True if valid
    
    Raises:
        ValueError: If invalid
    """
    if value < 0:
        raise ValueError(f"Invalid {name}: must be positive")
    return True


def sanitize_string(value: str, max_length: int = 255) -> str:
    """
    Sanitize a string input.
    
    - Strips whitespace
    - Truncates to max length
    - Removes null bytes
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
    
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        return str(value)
    
    # Remove null bytes
    value = value.replace("\x00", "")
    
    # Strip whitespace
    value = value.strip()
    
    # Truncate
    if len(value) > max_length:
        value = value[:max_length]
    
    return value


class ValidationError(Exception):
    """Custom validation error."""
    
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(message)
    
    def to_dict(self):
        result = {"message": self.message}
        if self.field:
            result["field"] = self.field
        return result
