"""
Unit tests for AirType API endpoints
"""

import pytest
import json
from unittest.mock import patch, MagicMock


class TestHealthEndpoints:
    """Tests for health check endpoints"""
    
    def test_health_check(self, client):
        """Test basic health check endpoint"""
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
    
    def test_detailed_health_check(self, client):
        """Test detailed health check endpoint"""
        response = client.get('/api/v1/health/detailed')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'database' in data
        assert 'redis' in data
        assert 'model' in data


class TestUserEndpoints:
    """Tests for user authentication endpoints"""
    
    def test_register_user(self, client):
        """Test user registration"""
        response = client.post('/api/v1/users/register', json={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'securepassword123'
        })
        
        assert response.status_code == 201
        data = response.get_json()
        assert 'message' in data
        assert data['message'] == 'User registered successfully'
    
    def test_register_duplicate_email(self, client, test_user):
        """Test registration with duplicate email"""
        response = client.post('/api/v1/users/register', json={
            'username': 'anotheruser',
            'email': 'test@example.com',
            'password': 'securepassword123'
        })
        
        assert response.status_code == 409
    
    def test_register_invalid_email(self, client):
        """Test registration with invalid email"""
        response = client.post('/api/v1/users/register', json={
            'username': 'newuser',
            'email': 'invalid-email',
            'password': 'securepassword123'
        })
        
        assert response.status_code == 400
    
    def test_login_success(self, client, test_user):
        """Test successful login"""
        response = client.post('/api/v1/users/login', json={
            'email': 'test@example.com',
            'password': 'testpassword123'
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'access_token' in data
        assert 'user' in data
    
    def test_login_invalid_credentials(self, client, test_user):
        """Test login with invalid credentials"""
        response = client.post('/api/v1/users/login', json={
            'email': 'test@example.com',
            'password': 'wrongpassword'
        })
        
        assert response.status_code == 401
    
    def test_login_nonexistent_user(self, client):
        """Test login with non-existent user"""
        response = client.post('/api/v1/users/login', json={
            'email': 'nonexistent@example.com',
            'password': 'password123'
        })
        
        assert response.status_code == 401
    
    def test_get_profile_authenticated(self, client, auth_headers):
        """Test getting user profile with authentication"""
        response = client.get('/api/v1/users/me', headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['email'] == 'test@example.com'
    
    def test_get_profile_unauthenticated(self, client):
        """Test getting user profile without authentication"""
        response = client.get('/api/v1/users/me')
        
        assert response.status_code == 401


class TestStrokeEndpoints:
    """Tests for stroke session endpoints"""
    
    def test_create_session(self, client, auth_headers):
        """Test creating a stroke session"""
        response = client.post('/api/v1/strokes/sessions', 
                               headers=auth_headers,
                               json={'device_info': {'browser': 'Chrome'}})
        
        assert response.status_code == 201
        data = response.get_json()
        assert 'session_id' in data
    
    def test_create_session_unauthenticated(self, client):
        """Test creating session without authentication"""
        response = client.post('/api/v1/strokes/sessions', json={})
        
        assert response.status_code == 401
    
    def test_add_stroke_points(self, client, auth_headers, sample_stroke_data):
        """Test adding points to a stroke session"""
        # First create a session
        create_response = client.post('/api/v1/strokes/sessions',
                                      headers=auth_headers,
                                      json={})
        session_id = create_response.get_json()['session_id']
        
        # Add points
        response = client.post(f'/api/v1/strokes/sessions/{session_id}/points',
                               headers=auth_headers,
                               json=sample_stroke_data)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['points_added'] == len(sample_stroke_data['points'])
    
    def test_complete_session(self, client, auth_headers, mock_model):
        """Test completing a stroke session"""
        # Create session
        create_response = client.post('/api/v1/strokes/sessions',
                                      headers=auth_headers,
                                      json={})
        session_id = create_response.get_json()['session_id']
        
        # Complete session
        response = client.post(f'/api/v1/strokes/sessions/{session_id}/complete',
                               headers=auth_headers)
        
        assert response.status_code == 200
    
    def test_get_session(self, client, auth_headers):
        """Test getting a stroke session"""
        # Create session
        create_response = client.post('/api/v1/strokes/sessions',
                                      headers=auth_headers,
                                      json={})
        session_id = create_response.get_json()['session_id']
        
        # Get session
        response = client.get(f'/api/v1/strokes/sessions/{session_id}',
                              headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['id'] == session_id


class TestPredictionEndpoints:
    """Tests for prediction endpoints"""
    
    def test_get_prediction_history(self, client, auth_headers):
        """Test getting prediction history"""
        response = client.get('/api/v1/predictions/history',
                              headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'predictions' in data
        assert 'total' in data
    
    def test_get_prediction_stats(self, client, auth_headers):
        """Test getting prediction statistics"""
        response = client.get('/api/v1/predictions/stats',
                              headers=auth_headers)
        
        assert response.status_code == 200
        data = response.get_json()
        assert 'period' in data


class TestValidation:
    """Tests for input validation"""
    
    def test_register_missing_fields(self, client):
        """Test registration with missing fields"""
        response = client.post('/api/v1/users/register', json={
            'username': 'newuser'
        })
        
        assert response.status_code == 400
    
    def test_register_short_password(self, client):
        """Test registration with short password"""
        response = client.post('/api/v1/users/register', json={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': '123'
        })
        
        assert response.status_code == 400
    
    def test_invalid_stroke_points(self, client, auth_headers):
        """Test adding invalid stroke points"""
        # Create session
        create_response = client.post('/api/v1/strokes/sessions',
                                      headers=auth_headers,
                                      json={})
        session_id = create_response.get_json()['session_id']
        
        # Add invalid points
        response = client.post(f'/api/v1/strokes/sessions/{session_id}/points',
                               headers=auth_headers,
                               json={'points': 'invalid'})
        
        assert response.status_code == 400
