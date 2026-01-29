-- Migration: Create training_samples table for user feedback and model training
-- Date: 2026-01-28
-- Purpose: Store stroke features with user corrections for continuous learning

CREATE TABLE IF NOT EXISTS training_samples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID,  -- Group related samples
    
    -- Feature data
    stroke_features JSONB NOT NULL,  -- 2D array: [[x, y, vx, vy, ax, ay, curvature], ...]
    stroke_metadata JSONB,           -- Optional: raw stroke points, timestamps, etc.
    
    -- Prediction data
    predicted_char VARCHAR(1) NOT NULL,
    predicted_confidence FLOAT NOT NULL,
    alternatives JSONB,              -- Top-k predictions with confidences
    
    -- Ground truth
    actual_char VARCHAR(1) NOT NULL,
    correction_type VARCHAR(20) DEFAULT 'manual',  -- 'manual', 'confirmed', 'corrected'
    
    -- Metadata
    model_version VARCHAR(50),
    inference_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CONSTRAINT valid_confidence CHECK (predicted_confidence >= 0 AND predicted_confidence <= 1),
    CONSTRAINT valid_chars CHECK (
        (predicted_char ~ '^[a-zA-Z0-9]$') AND 
        (actual_char ~ '^[a-zA-Z0-9]$')
    )
);

-- Indexes for efficient querying
CREATE INDEX idx_training_samples_user_id ON training_samples(user_id);
CREATE INDEX idx_training_samples_actual_char ON training_samples(actual_char);
CREATE INDEX idx_training_samples_predicted_char ON training_samples(predicted_char);
CREATE INDEX idx_training_samples_created_at ON training_samples(created_at DESC);
CREATE INDEX idx_training_samples_session_id ON training_samples(session_id);
CREATE INDEX idx_training_samples_confidence ON training_samples(predicted_confidence);

-- Index for finding misclassifications
CREATE INDEX idx_training_samples_misclassified ON training_samples(predicted_char, actual_char) 
WHERE predicted_char != actual_char;

-- GIN index for JSONB queries
CREATE INDEX idx_training_samples_features ON training_samples USING GIN (stroke_features);

-- Statistics view
CREATE OR REPLACE VIEW training_statistics AS
SELECT 
    actual_char,
    COUNT(*) as total_samples,
    COUNT(*) FILTER (WHERE predicted_char = actual_char) as correct_predictions,
    COUNT(*) FILTER (WHERE predicted_char != actual_char) as incorrect_predictions,
    ROUND(AVG(predicted_confidence)::numeric, 4) as avg_confidence,
    ROUND(
        (COUNT(*) FILTER (WHERE predicted_char = actual_char)::float / COUNT(*)::float * 100)::numeric, 
        2
    ) as accuracy_percentage
FROM training_samples
GROUP BY actual_char
ORDER BY total_samples DESC;

-- Create training session tracking table
CREATE TABLE IF NOT EXISTS training_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    model_version VARCHAR(50),
    samples_count INTEGER DEFAULT 0,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active',  -- 'active', 'completed', 'abandoned'
    
    CONSTRAINT valid_status CHECK (status IN ('active', 'completed', 'abandoned'))
);

CREATE INDEX idx_training_sessions_user_id ON training_sessions(user_id);
CREATE INDEX idx_training_sessions_status ON training_sessions(status);

-- Add comments for documentation
COMMENT ON TABLE training_samples IS 'Stores user corrections for continuous model improvement';
COMMENT ON COLUMN training_samples.stroke_features IS 'Normalized feature vectors extracted from strokes';
COMMENT ON COLUMN training_samples.correction_type IS 'Type of correction: manual (user corrected), confirmed (user confirmed correct)';
COMMENT ON VIEW training_statistics IS 'Per-character accuracy and confidence statistics';
