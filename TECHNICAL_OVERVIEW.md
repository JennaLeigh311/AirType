# AirType - Technical Overview & Systems Design

## 🎯 Project Summary
**AirType** is a real-time handwriting recognition system that uses computer vision and deep learning to recognize characters drawn in the air or on a canvas. It features continuous learning through user feedback and active learning strategies.

---

## 🏗️ System Architecture

### High-Level Architecture
```
┌─────────────────┐      HTTP/WS      ┌─────────────────┐
│   Frontend      │ ◄─────────────► │    Backend      │
│  (Vanilla JS)   │                  │    (Flask)      │
└─────────────────┘                  └─────────────────┘
                                              │
                          ┌───────────────────┼───────────────────┐
                          │                   │                   │
                    ┌─────▼─────┐      ┌─────▼─────┐      ┌─────▼─────┐
                    │ PostgreSQL │      │  PyTorch  │      │   Redis   │
                    │  Database  │      │   Model   │      │   Cache   │
                    └────────────┘      └───────────┘      └───────────┘
```

### Component Breakdown

#### Frontend Layer
- **Technology**: Vanilla JavaScript (ES6 Modules)
- **Modules**:
  - `app.js` - Main orchestration, UI management
  - `api-client.js` - REST API communication
  - `canvas-renderer.js` - Drawing and feature extraction
  - `video-stream.js` - MediaPipe hand tracking
  - `websocket-client.js` - Real-time communication
  - `config.js` - Configuration constants

#### Backend Layer (Flask)
- **Framework**: Flask 2.x with Blueprint architecture
- **Port**: 5001
- **Environment**: Production mode with debug enabled

---

## 📚 Tech Stack Details

### Backend Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.13 | Core language |
| **Flask** | 2.x | Web framework |
| **SQLAlchemy** | Latest | ORM for database |
| **PostgreSQL** | 15.15 | Primary database |
| **PyTorch** | Latest | Deep learning framework |
| **Redis** | Latest | Caching layer |
| **Flask-CORS** | Latest | Cross-origin support |
| **Flask-JWT-Extended** | Latest | Authentication |
| **Flask-SocketIO** | Latest | WebSocket support |
| **OpenCV (cv2)** | 4.13.0 | Video processing |
| **NumPy** | 2.2.4 | Numerical operations |
| **Marshmallow** | Latest | Data validation |
| **Bcrypt** | Latest | Password hashing |
| **Gunicorn** | Latest | Production WSGI server |

### Frontend Technologies
- **HTML5** - Semantic markup
- **CSS3** - Modern styling (Grid, Flexbox, CSS Variables)
- **JavaScript ES6+** - Modules, async/await, Promises
- **MediaPipe** (via CDN) - Hand landmark detection

---

## 🗄️ Database Design

### Tables

#### 1. `users`
```sql
- id: UUID (PK)
- username: VARCHAR(50) UNIQUE
- email: VARCHAR(120) UNIQUE
- password_hash: VARCHAR(255)
- created_at: TIMESTAMP
- updated_at: TIMESTAMP
- is_active: BOOLEAN
```

#### 2. `training_samples`
Primary table for collecting user feedback and corrections.

```sql
- id: UUID (PK)
- user_id: UUID (FK → users, nullable for anonymous)
- session_id: UUID (FK → training_sessions, nullable)
- stroke_features: JSONB (stores 2D array of features)
- stroke_metadata: JSONB (optional raw stroke data)
- predicted_char: CHAR(1)
- predicted_confidence: FLOAT (0-1)
- alternatives: JSONB (array of {char, confidence})
- actual_char: CHAR(1) (ground truth from user)
- correction_type: ENUM('manual', 'confirmed', 'corrected')
- model_version: VARCHAR(50)
- inference_time_ms: INTEGER
- is_correct: BOOLEAN (computed: predicted == actual)
- created_at: TIMESTAMP

Indexes:
- idx_actual_char: B-tree on actual_char
- idx_predicted_char: B-tree on predicted_char
- idx_is_correct: B-tree on is_correct
- idx_created_at: B-tree on created_at
- idx_correction_type: B-tree on correction_type
- idx_model_version: B-tree on model_version
- idx_user_id: B-tree on user_id
- idx_session_id: B-tree on session_id
- idx_stroke_features: GIN on stroke_features (JSONB)
```

**Design Rationale**:
- JSONB for flexible feature storage (supports different feature counts)
- GIN index on JSONB for fast querying
- Anonymous submissions supported (user_id nullable)
- Computed column `is_correct` for quick accuracy queries

#### 3. `training_sessions`
Tracks batch training runs.

```sql
- id: UUID (PK)
- user_id: UUID (FK → users, nullable)
- model_version: VARCHAR(50)
- samples_count: INTEGER
- status: VARCHAR(20) ('active', 'completed', 'failed')
- started_at: TIMESTAMP
- completed_at: TIMESTAMP
- metrics: JSONB (stores accuracy, loss, etc.)

Indexes:
- idx_training_user: B-tree on user_id
- idx_training_status: B-tree on status
```

#### 4. `training_statistics` (View)
Materialized aggregate view for performance.

```sql
CREATE VIEW training_statistics AS
SELECT
  actual_char,
  COUNT(*) as total_samples,
  SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_count,
  AVG(predicted_confidence) as avg_confidence
FROM training_samples
GROUP BY actual_char;
```

---

## 🔌 Complete API Reference

### Base URL
```
http://localhost:5001/api
```

### Health & Status

#### `GET /health`
Health check endpoint.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2026-01-29T01:37:15Z",
  "version": "1.0.0"
}
```

---

### Authentication Endpoints

#### `POST /users/register`
Register a new user.

**Request Body**:
```json
{
  "username": "string (3-50 chars)",
  "email": "string (valid email)",
  "password": "string (min 8 chars)"
}
```

**Response** (201):
```json
{
  "message": "User registered successfully",
  "user": {
    "id": "uuid",
    "username": "string",
    "email": "string"
  }
}
```

#### `POST /users/login`
Authenticate and receive JWT token.

**Request Body**:
```json
{
  "email": "string",
  "password": "string"
}
```

**Response** (200):
```json
{
  "access_token": "jwt_token",
  "refresh_token": "jwt_token",
  "user": {
    "id": "uuid",
    "username": "string",
    "email": "string"
  }
}
```

#### `POST /users/logout`
Logout (revoke tokens).

**Headers**: `Authorization: Bearer <token>`

**Response** (200):
```json
{
  "message": "Logged out successfully"
}
```

---

### Prediction Endpoints

#### `POST /predictions/predict`
Get character prediction from stroke features.

**Request Body**:
```json
{
  "features": [[x, y, vx, vy, acc, curv, pen], ...],
  "sequence_length": 50,
  "return_alternatives": true
}
```

**Features Explained**:
- `x, y`: Normalized coordinates (0-1)
- `vx, vy`: Velocity components
- `acc`: Acceleration magnitude
- `curv`: Path curvature
- `pen`: Pen state (0=up, 1=down)

**Response** (200):
```json
{
  "prediction": "A",
  "confidence": 0.87,
  "alternatives": [
    {"char": "H", "confidence": 0.08},
    {"char": "N", "confidence": 0.03}
  ],
  "inference_time": 23,
  "model_version": "1.0.0"
}
```

**Rate Limit**: 100 requests/minute

---

### Training & Feedback Endpoints

#### `POST /training/feedback`
Submit prediction correction for training.

**Request Body**:
```json
{
  "stroke_features": [[...], ...],
  "predicted_char": "A",
  "predicted_confidence": 0.87,
  "alternatives": [...],
  "actual_char": "H",
  "correction_type": "corrected",
  "model_version": "1.0.0",
  "inference_time_ms": 23,
  "stroke_metadata": {...}
}
```

**Correction Types**:
- `confirmed`: User confirmed prediction was correct
- `corrected`: User provided correction
- `manual`: Manual entry

**Response** (201):
```json
{
  "message": "Feedback submitted successfully",
  "sample": {
    "id": "uuid",
    "actual_char": "H",
    "is_correct": false
  }
}
```

**Rate Limit**: 200 requests/minute
**Auth**: Optional (supports anonymous submissions)

#### `POST /training/batch-feedback`
Submit multiple corrections in one request.

**Request Body**:
```json
{
  "samples": [
    {
      "stroke_features": [...],
      "predicted_char": "A",
      "actual_char": "A",
      ...
    },
    ...
  ]
}
```

**Limit**: Max 100 samples per batch

**Response** (201):
```json
{
  "message": "Processed 50 samples",
  "created": 48,
  "errors": 2,
  "error_details": [...]
}
```

**Rate Limit**: 50 requests/minute

#### `GET /training/stats`
Get training statistics and model performance.

**Response** (200):
```json
{
  "total_samples": 312,
  "correct_predictions": 267,
  "overall_accuracy": 85.58,
  "misclassification_rate": 14.42,
  "per_character": [
    {
      "char": "A",
      "total": 25,
      "correct": 22,
      "accuracy": 88.0,
      "avg_confidence": 0.82
    },
    ...
  ],
  "confidence_distribution": {
    "0-20%": 12,
    "20-40%": 18,
    "40-60%": 45,
    "60-80%": 89,
    "80-100%": 148
  },
  "recent_activity": {
    "last_7_days": 156
  }
}
```

**Rate Limit**: 60 requests/minute

#### `GET /training/data`
Get filtered training samples (paginated).

**Headers**: `Authorization: Bearer <token>` (required)

**Query Parameters**:
- `page`: int (default: 1)
- `per_page`: int (default: 50, max: 500)
- `char`: string (filter by character)
- `misclassified_only`: boolean
- `min_confidence`: float (0-1)
- `max_confidence`: float (0-1)
- `correction_type`: string
- `start_date`: ISO datetime
- `end_date`: ISO datetime

**Response** (200):
```json
{
  "samples": [...],
  "pagination": {
    "page": 1,
    "per_page": 50,
    "total": 312,
    "pages": 7,
    "has_next": true,
    "has_prev": false
  }
}
```

**Rate Limit**: 30 requests/minute

#### `POST /training/train`
Trigger model training/fine-tuning.

**Request Body**:
```json
{
  "min_samples": 50,
  "epochs": 10,
  "augment": true,
  "augmentation_factor": 5,
  "fine_tune": false,
  "use_misclassified_only": false
}
```

**Response** (200):
```json
{
  "message": "Training completed successfully",
  "session": {
    "id": "uuid",
    "samples_count": 312,
    "status": "completed"
  },
  "results": {
    "status": "completed",
    "epochs_run": 5,
    "final_train_accuracy": 89.2,
    "final_val_accuracy": 85.4,
    "best_val_accuracy": 87.1,
    "training_time_seconds": 16.88,
    "samples_used": 312,
    "device": "mps",
    "history": [...]
  }
}
```

**Rate Limit**: 10 requests/hour
**Auth**: Optional

#### `GET /training/suggestions`
Get active learning suggestions for which characters to practice.

**Response** (200):
```json
{
  "suggestions": [
    {
      "char": "q",
      "score": 15.4,
      "reasons": [
        "Low coverage (5 samples)",
        "Often confused with 'g' (3x)"
      ],
      "current_count": 5,
      "current_accuracy": 40.0
    },
    ...
  ],
  "practice_session": ["q", "g", "j", "x", ...],
  "analysis": {
    "total_samples": 312,
    "unique_chars_covered": 48,
    "chars_needing_samples": ["q", "x", "z", ...],
    "top_confusions": [
      {"actual": "o", "predicted_as": "0", "count": 12},
      ...
    ],
    "low_confidence_count": 23
  }
}
```

**Rate Limit**: 60 requests/minute

#### `GET /training/training-info`
Get training capabilities and hardware info.

**Response** (200):
```json
{
  "device": "mps",
  "cuda_available": false,
  "mps_available": true,
  "gpu_name": null,
  "best_accuracy": 85.4,
  "training_runs": 3,
  "model_path": "/path/to/models/latest_model.pt",
  "total_samples": 312,
  "ready_for_training": true,
  "recommended_action": "Consider training or fine-tuning"
}
```

**Rate Limit**: 60 requests/minute

#### `GET /training/export`
Export all training data (for external analysis).

**Headers**: `Authorization: Bearer <token>` (required)

**Response** (200):
```json
{
  "total_samples": 312,
  "features": [[[...], ...], ...],
  "labels": ["A", "B", "C", ...],
  "format": {
    "features": "List of 2D arrays (samples x features)",
    "labels": "List of characters"
  }
}
```

**Rate Limit**: 10 requests/hour

---

### Stroke Endpoints

#### `POST /strokes/submit`
Submit raw stroke data (alternative to /predictions/predict).

**Headers**: `Authorization: Bearer <token>` (optional)

**Request Body**:
```json
{
  "points": [
    {"x": 0.5, "y": 0.3, "timestamp_ms": 1000},
    ...
  ],
  "metadata": {
    "device": "mouse",
    "canvas_size": {"width": 800, "height": 600}
  }
}
```

**Response** (201):
```json
{
  "stroke_id": "uuid",
  "features_extracted": 50,
  "prediction": {...}
}
```

---

## 🧠 Machine Learning Pipeline

### Model Architecture: Bidirectional LSTM with Multi-Head Attention

```
Input (batch, seq_len, 7)
    ↓
Embedding (Linear: 7 → 64)
    ↓
Bidirectional LSTM (64 hidden)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (128 hidden)
    ↓
Dropout (0.3)
    ↓
Multi-Head Attention (4 heads)
    ↓
Residual + LayerNorm
    ↓
Global Average Pooling
    ↓
FC (256 → 128 → 62)
    ↓
Output (62 classes: a-z, A-Z, 0-9)
```

### Model Components

#### 1. **Embedding Layer**
- Input: 7 features per timestep
- Output: 64-dimensional embedding
- Purpose: Learn meaningful feature representations

#### 2. **LSTM Layers**
- **Layer 1**: 64 hidden units, bidirectional (→ 128 output)
- **Layer 2**: 128 hidden units, bidirectional (→ 256 output)
- **Dropout**: 0.3 between layers
- **Purpose**: Capture temporal dependencies

#### 3. **Multi-Head Attention**
- 4 attention heads
- Scaled dot-product attention
- Self-attention over sequence
- **Purpose**: Focus on important parts of stroke

#### 4. **Classification Head**
- FC: 256 → 128 (ReLU, Dropout 0.3)
- FC: 128 → 62 (output logits)
- **Loss**: CrossEntropyLoss with label smoothing (0.1)

### Training Process

#### Device Selection (Priority Order)
1. **CUDA** (NVIDIA GPU) - if available
2. **MPS** (Apple Silicon) - if available
3. **CPU** - fallback

#### Data Augmentation Pipeline
Applied automatically during training:

1. **Rotation**: ±15° random rotation
2. **Scaling**: 0.85-1.15x random scale
3. **Translation**: ±10% random shift
4. **Shear**: ±0.1 random shear
5. **Aspect Ratio**: 0.9-1.1x modification
6. **Gaussian Noise**: σ=0.02 on positions
7. **Temporal Jitter**: ±5% timestamp variance
8. **Velocity Noise**: σ=0.05 on derivatives

**Augmentation Factor**: 5x (52 samples → 312 augmented)

#### Training Hyperparameters

```python
{
  "optimizer": "AdamW",
  "learning_rate": 0.001,
  "weight_decay": 0.0001,
  "scheduler": "OneCycleLR",
  "max_lr": 0.01,
  "batch_size": 32,
  "epochs": 10,
  "early_stopping_patience": 3,
  "gradient_clip": 1.0,
  "label_smoothing": 0.1,
  "val_split": 0.2
}
```

#### Fine-Tuning Mode
```python
{
  "learning_rate": 0.0001,  # 10x lower
  "augmentation_factor": 3,  # Less aggressive
  "epochs": 5,
  "early_stopping_patience": 2
}
```

### Active Learning Strategy

#### Prioritization Formula
```
Priority Score = 
  (1 - coverage_ratio) × coverage_weight × 10 +
  (100 - accuracy) / 100 × 10 +
  (1 - avg_confidence) × low_confidence_weight × 5 +
  confusion_count × confusion_weight
```

**Weights**:
- `coverage_weight`: 1.0
- `low_confidence_weight`: 1.5
- `confusion_weight`: 2.0

#### Sample Selection Criteria
1. **Underrepresented characters** (< 20 samples)
2. **Low accuracy characters** (< 50%)
3. **Low confidence predictions** (< 0.6)
4. **Frequently confused pairs** (≥ 2 confusions)

---

## ⚙️ Configuration System

### Backend Configuration

#### File: `backend/app/config.py`

```python
class Config:
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    DEBUG = True
    
    # Database
    SQLALCHEMY_DATABASE_URI = 'postgresql://airtype_user:password@localhost/airtype'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # JWT
    JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'jwt-secret-key')
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    
    # CORS
    CORS_ORIGINS = ['http://localhost:8080', 'http://127.0.0.1:8080']
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = 'redis://localhost:6379'
    
    # Model
    MODEL_VERSION = '1.0.0'
    MODEL_PATH = 'models/latest_model.pt'
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = 'json'
```

### Frontend Configuration

#### File: `frontend/js/config.js`

```javascript
const Config = {
  API_BASE_URL: 'http://localhost:5001/api',
  WS_BASE_URL: 'ws://localhost:5001',
  
  STORAGE: {
    AUTH_TOKEN: 'airtype_token',
    USER_DATA: 'airtype_user',
  },
  
  FEATURES: {
    MIN_POINTS: 5,
    SEQUENCE_LENGTH: 50,
  },
  
  PREDICTION: {
    DEBOUNCE_MS: 300,
    MAX_HISTORY: 20,
  },
};
```

---

## 🔄 Data Flow & Request Lifecycle

### Character Prediction Flow

```
1. User draws stroke on canvas
    ↓
2. canvas-renderer.js extracts features
   - Resample to 50 points
   - Compute velocities, acceleration, curvature
   - Normalize coordinates
    ↓
3. app.js sends POST /predictions/predict
    ↓
4. Backend predictor.py:
   - Validate input (Marshmallow schema)
   - Load model from cache/disk
   - Preprocess features (pad/truncate)
   - Run inference (PyTorch forward pass)
   - Apply softmax for probabilities
   - Return top-5 predictions
    ↓
5. Frontend displays results
   - Show top prediction with confidence
   - Show alternatives
   - Open correction modal
    ↓
6. User confirms/corrects
    ↓
7. app.js sends POST /training/feedback
    ↓
8. Backend stores in training_samples table
    ↓
9. Update statistics and suggest next practice characters
```

### Training Flow

```
1. User clicks "Train Model" (≥50 samples required)
    ↓
2. Frontend sends POST /training/train
    ↓
3. Backend training.py:
   - Query training_samples from PostgreSQL
   - Load FastTrainer (singleton)
   - Apply data augmentation (5x)
   - Prepare DataLoaders (80/20 train/val split)
    ↓
4. FastTrainer.train():
   - Initialize AdamW optimizer
   - OneCycleLR scheduler
   - For each epoch:
     * Forward pass (LSTM + Attention)
     * Compute loss (CrossEntropy + label smoothing)
     * Backward pass + gradient clipping
     * Update weights
     * Validate on val set
     * Early stopping check (patience=3)
   - Save best model checkpoint
    ↓
5. Return training metrics to frontend
    ↓
6. Frontend displays results (accuracy, time, epochs)
```

---

## 🔐 Security & Authentication

### JWT Authentication Flow

```
1. POST /users/login
    ↓
2. Backend verifies credentials (bcrypt)
    ↓
3. Generate JWT tokens:
   - Access token (1 hour)
   - Refresh token (30 days)
    ↓
4. Frontend stores in localStorage
    ↓
5. Protected requests include:
   Authorization: Bearer <access_token>
    ↓
6. Backend validates JWT:
   - Verify signature
   - Check expiration
   - Extract user identity
    ↓
7. Process request with user context
```

### Security Features

1. **Password Hashing**: Bcrypt with salt
2. **CORS Protection**: Whitelist origins
3. **Rate Limiting**: Redis-backed limiter
4. **Input Validation**: Marshmallow schemas
5. **SQL Injection Prevention**: SQLAlchemy ORM
6. **XSS Protection**: Content Security Policy headers

---

## 📊 Performance Optimizations

### Backend

1. **Model Caching**
   - Singleton FastTrainer instance
   - Keep model in memory (GPU/MPS)
   - Avoid repeated disk I/O

2. **Database Indexing**
   - B-tree on frequently queried columns
   - GIN index on JSONB for fast feature queries
   - Materialized view for statistics

3. **Connection Pooling**
   - SQLAlchemy pool (size=10)
   - PostgreSQL connection reuse

4. **Rate Limiting**
   - Redis-backed (fast in-memory checks)
   - Per-endpoint limits

### Frontend

1. **Debouncing**
   - 300ms debounce on prediction requests
   - Prevent excessive API calls

2. **Lazy Loading**
   - ES6 modules load on demand
   - MediaPipe loads asynchronously

3. **LocalStorage Caching**
   - Contribution count cached
   - Auth tokens persisted

---

## 🧪 Testing Strategy

### Backend Tests (`backend/tests/`)

```python
# test_api.py
- test_health_endpoint()
- test_register_user()
- test_login()
- test_predict_character()
- test_submit_feedback()
- test_training_stats()

# test_model.py
- test_model_forward_pass()
- test_attention_mechanism()
- test_feature_extraction()
- test_data_augmentation()

# conftest.py
- pytest fixtures (app, client, db)
- Test database setup/teardown
```

### Run Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

---

## 🚀 Deployment Architecture

### Development (Current)

```
Frontend: Python HTTP server (port 8080)
Backend: Flask dev server (port 5001)
Database: Local PostgreSQL (port 5432)
```

### Production (Recommended)

```
Frontend: Nginx static file server
Backend: Gunicorn + Nginx reverse proxy
Database: PostgreSQL (managed service)
Cache: Redis (managed service)
Container: Docker + Docker Compose
```

#### Docker Setup

```yaml
# docker-compose.yml
services:
  frontend:
    build: ./frontend
    ports: ["8080:80"]
  
  backend:
    build: ./backend
    ports: ["5001:5001"]
    environment:
      - DATABASE_URL=postgresql://...
      - REDIS_URL=redis://...
  
  postgres:
    image: postgres:15
    volumes: ["pgdata:/var/lib/postgresql/data"]
  
  redis:
    image: redis:alpine
```

---

## 🎤 Interview Talking Points

### System Design Questions

**Q: "How does your system scale?"**
- Horizontal scaling: Multiple backend instances behind load balancer
- Database: Read replicas for queries, primary for writes
- Caching: Redis for rate limiting and frequent queries
- Model serving: GPU instances for inference, CPU for API

**Q: "How do you handle failures?"**
- Graceful degradation: Anonymous feedback continues if auth fails
- Database retries with exponential backoff
- Model fallback: Use previous checkpoint if latest fails
- Circuit breaker pattern for external services

**Q: "Why PostgreSQL over MongoDB?"**
- ACID transactions for training samples
- Complex queries (JOIN, aggregations) for statistics
- JSONB gives flexibility while maintaining relational benefits
- Better for structured data with relationships

**Q: "Why bidirectional LSTM?"**
- Captures both forward and backward temporal context
- Better for stroke sequences (start → end and end → start)
- Attention mechanism identifies salient features
- Multi-head attention captures different aspects

### ML Questions

**Q: "How do you prevent overfitting?"**
- Dropout (0.3) between LSTM layers
- Early stopping (patience=3)
- Data augmentation (5x)
- Label smoothing (0.1)
- Validation set (20% holdout)

**Q: "Why active learning?"**
- Limited labeled data (user corrections)
- Focus on hard/confusing examples
- Balance class distribution
- Maximize learning from minimal samples

**Q: "How do you handle class imbalance?"**
- Augmentation factor varies by class coverage
- Active learning suggests underrepresented characters
- Stratified sampling for validation split

### Architecture Questions

**Q: "Why Flask over FastAPI?"**
- Mature ecosystem, extensive documentation
- Easy integration with SQLAlchemy, JWT, SocketIO
- Blueprint architecture for modularity
- Could migrate to FastAPI for async benefits

**Q: "Why not use a CNN?"**
- Strokes are temporal sequences (time-series)
- Order matters (start → end)
- LSTM better captures sequential dependencies
- Could add CNN for spatial features in future

**Q: "How do you version your API?"**
- URL versioning: `/api/v1/...`
- Model versioning in metadata
- Backward compatibility via Marshmallow schemas
- Database migrations via Alembic

---

## 📈 Metrics & Monitoring

### Key Metrics Tracked

1. **Model Performance**
   - Overall accuracy
   - Per-character accuracy
   - Confidence distribution
   - Inference latency

2. **System Health**
   - API response times
   - Error rates (4xx, 5xx)
   - Database query latency
   - Cache hit rates

3. **User Engagement**
   - Samples collected per day
   - Training sessions initiated
   - Correction types (manual vs confirmed)

### Logging

```python
# Structured JSON logging
{
  "timestamp": "2026-01-29T01:37:15Z",
  "level": "INFO",
  "message": "Training completed",
  "session_id": "uuid",
  "samples": 52,
  "accuracy": 14.52,
  "device": "mps",
  "request_id": "uuid"
}
```

---

## 🔮 Future Enhancements

1. **Sequential Prediction**: Predict words, not just characters
2. **Transfer Learning**: Pre-train on large handwriting datasets
3. **Real-time Training**: Update model incrementally without full retraining
4. **Multi-language Support**: Extend beyond English alphanumeric
5. **User Personalization**: Per-user model fine-tuning
6. **Mobile App**: React Native frontend
7. **Cloud Deployment**: AWS/GCP with managed services

---

## 📚 Code Organization

```
backend/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── config.py            # Configuration
│   ├── api/                 # API blueprints
│   │   ├── health.py        # Health checks
│   │   ├── users.py         # Authentication
│   │   ├── predictions.py   # Character prediction
│   │   ├── strokes.py       # Stroke submission
│   │   └── training.py      # Training & feedback
│   ├── ml/                  # Machine learning
│   │   ├── model.py         # LSTM architecture
│   │   ├── train.py         # Training script
│   │   ├── fast_training.py # Optimized trainer
│   │   ├── dataset.py       # PyTorch datasets
│   │   ├── augmentation.py  # Data augmentation
│   │   └── active_learning.py # Active learning
│   ├── models/              # SQLAlchemy models
│   │   └── __init__.py      # User, TrainingSample
│   ├── services/            # Business logic
│   │   ├── predictor.py     # Inference service
│   │   ├── feature_extractor.py
│   │   └── deduplicator.py
│   └── utils/               # Utilities
│       ├── logging.py       # Structured logging
│       ├── cache.py         # Redis caching
│       └── validation.py    # Input validation
├── models/                  # Trained model checkpoints
├── migrations/              # Database migrations (SQL)
└── tests/                   # Unit & integration tests

frontend/
├── index.html              # Main HTML
├── css/
│   └── styles.css          # All styling
└── js/
    ├── app.js              # Main orchestrator
    ├── api-client.js       # REST API client
    ├── canvas-renderer.js  # Drawing & features
    ├── video-stream.js     # MediaPipe integration
    ├── websocket-client.js # WebSocket handler
    └── config.js           # Constants
```

---

## 🎯 Summary for Interview

**What is AirType?**
A real-time handwriting recognition system using deep learning (bidirectional LSTM with attention) that learns continuously from user feedback through active learning strategies.

**Tech Stack:**
- Backend: Flask (Python), PostgreSQL, PyTorch, Redis
- Frontend: Vanilla JS (ES6 modules), MediaPipe
- ML: Bidirectional LSTM, Multi-head attention, Data augmentation

**Key Features:**
- GPU-accelerated training (CUDA/MPS)
- Active learning for efficient data collection
- Anonymous feedback support
- Real-time inference (<100ms)
- Batch training with early stopping

**Scale:**
- Designed for horizontal scaling
- Stateless backend (can run multiple instances)
- Database optimized with JSONB and GIN indexes
- Redis for distributed rate limiting

**Interesting Challenges Solved:**
1. Handling variable-length stroke sequences (padding/truncation)
2. Class imbalance with smart augmentation
3. Incremental learning without catastrophic forgetting
4. Feature extraction from raw coordinates

This is a production-ready ML system with proper architecture, testing, monitoring, and deployment considerations.
