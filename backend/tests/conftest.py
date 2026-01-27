"""
Test configuration and fixtures for AirType backend tests
"""

import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch

# Set test environment before importing app
os.environ['FLASK_ENV'] = 'testing'
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
os.environ['SECRET_KEY'] = 'test-secret-key'
os.environ['JWT_SECRET_KEY'] = 'test-jwt-secret'

from app import create_app, db
from app.models import User, StrokeSession, Prediction


@pytest.fixture(scope='function')
def app():
    """Create application for testing"""
    # Create temp directory for test files
    test_dir = tempfile.mkdtemp()
    
    app = create_app('testing')
    app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'WTF_CSRF_ENABLED': False,
        'JWT_SECRET_KEY': 'test-jwt-secret',
        'MODEL_PATH': os.path.join(test_dir, 'test_model.pt')
    })
    
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()


@pytest.fixture(scope='function')
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture(scope='function')
def db_session(app):
    """Create database session for testing"""
    with app.app_context():
        yield db.session


@pytest.fixture
def test_user(app, db_session):
    """Create a test user"""
    with app.app_context():
        user = User(
            username='testuser',
            email='test@example.com'
        )
        user.set_password('testpassword123')
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)
        return user


@pytest.fixture
def auth_headers(app, test_user, client):
    """Get authentication headers for test user"""
    response = client.post('/api/v1/users/login', json={
        'email': 'test@example.com',
        'password': 'testpassword123'
    })
    
    data = response.get_json()
    token = data.get('access_token')
    
    return {'Authorization': f'Bearer {token}'}


@pytest.fixture
def mock_model():
    """Mock the ML model"""
    with patch('app.services.predictor.predictor') as mock:
        mock.predict.return_value = {
            'prediction': 'A',
            'confidence': 0.95,
            'alternatives': [
                {'char': 'H', 'confidence': 0.03},
                {'char': 'N', 'confidence': 0.02}
            ],
            'inference_time': 45.5
        }
        yield mock


@pytest.fixture
def sample_stroke_data():
    """Sample stroke data for testing"""
    return {
        'points': [
            {'x': 0.1, 'y': 0.1, 'timestamp': 1000},
            {'x': 0.2, 'y': 0.2, 'timestamp': 1050},
            {'x': 0.3, 'y': 0.3, 'timestamp': 1100},
            {'x': 0.4, 'y': 0.4, 'timestamp': 1150},
            {'x': 0.5, 'y': 0.5, 'timestamp': 1200},
            {'x': 0.6, 'y': 0.4, 'timestamp': 1250},
            {'x': 0.7, 'y': 0.3, 'timestamp': 1300},
            {'x': 0.8, 'y': 0.2, 'timestamp': 1350},
            {'x': 0.9, 'y': 0.1, 'timestamp': 1400},
            {'x': 1.0, 'y': 0.0, 'timestamp': 1450}
        ]
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    with patch('app.utils.cache.redis_client') as mock:
        mock.get.return_value = None
        mock.set.return_value = True
        mock.delete.return_value = True
        yield mock
