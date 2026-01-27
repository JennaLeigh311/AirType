#!/usr/bin/env python
"""
Quick setup validation script for AirType backend
Tests that all critical imports work and configuration is valid
"""

import sys
import os

# Set environment for testing
os.environ['FLASK_ENV'] = 'development'
os.environ['DATABASE_URL'] = 'sqlite:///test.db'
os.environ['REDIS_URL'] = 'redis://localhost:6379/0'
os.environ['SECRET_KEY'] = 'test-secret-key'
os.environ['JWT_SECRET_KEY'] = 'test-jwt-secret'

print("🔍 Testing AirType Backend Setup...\n")

# Test 1: Core imports
print("1️⃣  Testing core imports...")
try:
    import flask
    import flask_sqlalchemy
    import flask_jwt_extended
    import flask_socketio
    import torch
    import numpy as np
    import cv2
    print("   ✅ All core packages imported successfully")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test 2: MediaPipe
print("\n2️⃣  Testing MediaPipe...")
try:
    import mediapipe as mp
    print(f"   ✅ MediaPipe {mp.__version__} imported successfully")
except ImportError as e:
    print(f"   ❌ MediaPipe import error: {e}")
    sys.exit(1)

# Test 3: PyTorch
print("\n3️⃣  Testing PyTorch...")
try:
    print(f"   ✅ PyTorch {torch.__version__}")
    print(f"   ✅ MPS (Apple Silicon GPU) available: {torch.backends.mps.is_available()}")
    print(f"   ✅ CPU threads: {torch.get_num_threads()}")
except Exception as e:
    print(f"   ⚠️  PyTorch test failed: {e}")

# Test 4: Flask app creation
print("\n4️⃣  Testing Flask app creation...")
try:
    from app import create_app
    app = create_app()
    print(f"   ✅ Flask app created successfully")
    print(f"   ✅ Debug mode: {app.debug}")
    print(f"   ✅ Testing mode: {app.testing}")
except Exception as e:
    print(f"   ❌ Flask app creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Database models
print("\n5️⃣  Testing database models...")
try:
    from app.models import User, StrokeSession, StrokePoint, Prediction
    print("   ✅ All database models imported successfully")
except Exception as e:
    print(f"   ❌ Model import failed: {e}")
    sys.exit(1)

# Test 6: ML model architecture
print("\n6️⃣  Testing ML model architecture...")
try:
    from app.ml.model import HandwritingLSTM
    model = HandwritingLSTM(
        input_size=7,
        hidden_size=128,
        num_layers=2,
        num_classes=62
    )
    print(f"   ✅ LSTM model created successfully")
    
    # Test forward pass
    test_input = torch.randn(1, 50, 7)
    with torch.no_grad():
        output = model(test_input)
    print(f"   ✅ Forward pass successful: output shape {output.shape}")
except Exception as e:
    print(f"   ❌ ML model test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 7: Services
print("\n7️⃣  Testing services...")
try:
    from app.services.feature_extractor import FeatureExtractor
    from app.services.deduplicator import Deduplicator
    
    extractor = FeatureExtractor()
    dedup = Deduplicator()
    print("   ✅ Services initialized successfully")
except Exception as e:
    print(f"   ⚠️  Service initialization warning: {e}")

# Test 8: API blueprints
print("\n8️⃣  Testing API blueprints...")
try:
    from app.api import health_bp, strokes_bp, predictions_bp, users_bp
    print("   ✅ All API blueprints imported successfully")
except Exception as e:
    print(f"   ❌ Blueprint import failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✅ Setup validation complete! Backend is ready to run.")
print("="*60)
print("\n📝 Next steps:")
print("   1. Start PostgreSQL: brew services start postgresql")
print("   2. Start Redis: brew services start redis")
print("   3. Run backend: python wsgi.py")
print("   4. Or use Docker: cd .. && docker-compose up -d")
print()
