"""
AirType Predictor Service

This module provides model inference for handwriting character prediction
with caching, batch processing, and performance monitoring.
"""

from typing import Dict, List, Optional, Any
import os
import time
import logging
from pathlib import Path

import numpy as np
import torch
import redis

from app.ml.model import HandwritingLSTM
from app.services.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class Predictor:
    """
    Handles model inference for character prediction.
    
    Features:
    - Model loading and caching
    - Batch prediction support
    - Redis caching for repeated predictions
    - Performance monitoring
    """
    
    _instance: Optional["Predictor"] = None
    _model: Optional[HandwritingLSTM] = None
    _device: str = "cpu"
    _model_version: str = "1.0.0"
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for model reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        redis_url: Optional[str] = None,
        cache_ttl: int = 3600,
    ):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('auto', 'cuda', 'cpu', 'mps')
            redis_url: Redis URL for caching
            cache_ttl: Cache TTL in seconds
        """
        # Only initialize once (singleton)
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self._initialized = True
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device
        
        # Model path
        self.model_path = model_path or os.getenv(
            "MODEL_PATH",
            str(Path(__file__).parent.parent / "ml" / "checkpoints" / "best_model.pth"),
        )
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            sequence_length=50,
            normalize=True,
        )
        
        # Redis caching
        self.redis_client: Optional[redis.Redis] = None
        self.cache_ttl = cache_ttl
        
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Caching disabled.")
                self.redis_client = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model from disk."""
        try:
            if os.path.exists(self.model_path):
                self._model = HandwritingLSTM.load_pretrained(
                    self.model_path,
                    device=self._device,
                )
                self._model.eval()
                logger.info(f"Model loaded from {self.model_path}")
            else:
                # Create dummy model for development
                logger.warning(f"Model not found at {self.model_path}. Using untrained model.")
                self._model = HandwritingLSTM()
                self._model.to(self._device)
                self._model.eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._model = HandwritingLSTM()
            self._model.to(self._device)
            self._model.eval()
    
    def is_model_loaded(self) -> bool:
        """Check if a trained model is loaded."""
        return self._model is not None and os.path.exists(self.model_path)
    
    def predict(
        self,
        features: np.ndarray,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Make a prediction from feature array.
        
        Args:
            features: Feature array of shape (seq_len, 7)
            top_k: Number of top predictions to return
        
        Returns:
            Dictionary with prediction results
        """
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = self._get_cache_key(features)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result is not None:
                cached_result["from_cache"] = True
                return cached_result
            
            # Prepare input tensor
            if len(features.shape) == 2:
                features = features[np.newaxis, ...]  # Add batch dimension
            
            features_tensor = torch.FloatTensor(features).to(self._device)
            
            # Run inference
            with torch.no_grad():
                results = self._model.predict(
                    features_tensor,
                    top_k=top_k,
                )
            
            result = results[0]  # Get first (only) batch item
            
            # Add inference time
            inference_time_ms = int((time.time() - start_time) * 1000)
            result["inference_time_ms"] = inference_time_ms
            result["from_cache"] = False
            result["model_version"] = self._model_version
            
            # Cache result
            self._set_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            inference_time_ms = int((time.time() - start_time) * 1000)
            
            return {
                "predicted_char": "?",
                "confidence": 0.0,
                "alternatives": [],
                "inference_time_ms": inference_time_ms,
                "error": str(e),
            }
    
    def predict_from_points(
        self,
        points: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Make a prediction from raw point data.
        
        Args:
            points: List of point dictionaries
            top_k: Number of top predictions
        
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.feature_extractor.extract_sequence_features(points)
        
        return self.predict(features, top_k=top_k)
    
    def predict_from_session(
        self,
        session_id: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Make a prediction from a stroke session.
        
        Args:
            session_id: UUID of the stroke session
            top_k: Number of top predictions
        
        Returns:
            Dictionary with prediction results
        """
        from app.models import StrokeSession, StrokePoint
        
        # Get session and points
        session = StrokeSession.query.get(session_id)
        
        if session is None:
            raise ValueError(f"Session not found: {session_id}")
        
        points = session.stroke_points.order_by(
            StrokePoint.sequence_number
        ).all()
        
        if len(points) < 5:
            raise ValueError(f"Session has insufficient points: {len(points)}")
        
        # Convert to dictionaries
        point_dicts = [p.to_dict() for p in points]
        
        return self.predict_from_points(point_dicts, top_k=top_k)
    
    def predict_batch(
        self,
        features_batch: List[np.ndarray],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of feature arrays.
        
        Args:
            features_batch: List of feature arrays
            top_k: Number of top predictions
        
        Returns:
            List of prediction result dictionaries
        """
        start_time = time.time()
        
        try:
            # Pad sequences to same length
            max_len = max(f.shape[0] for f in features_batch)
            batch_size = len(features_batch)
            
            padded = np.zeros((batch_size, max_len, 7))
            lengths = []
            masks = np.zeros((batch_size, max_len))
            
            for i, features in enumerate(features_batch):
                seq_len = features.shape[0]
                padded[i, :seq_len] = features
                lengths.append(seq_len)
                masks[i, :seq_len] = 1
            
            # Convert to tensors
            features_tensor = torch.FloatTensor(padded).to(self._device)
            lengths_tensor = torch.LongTensor(lengths)
            mask_tensor = torch.FloatTensor(masks).to(self._device)
            
            # Run inference
            with torch.no_grad():
                results = self._model.predict(
                    features_tensor,
                    lengths=lengths_tensor,
                    mask=mask_tensor,
                    top_k=top_k,
                )
            
            # Add timing to results
            total_time = time.time() - start_time
            for i, result in enumerate(results):
                result["inference_time_ms"] = int(total_time * 1000 / batch_size)
                result["model_version"] = self._model_version
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}", exc_info=True)
            
            return [
                {
                    "predicted_char": "?",
                    "confidence": 0.0,
                    "alternatives": [],
                    "error": str(e),
                }
                for _ in features_batch
            ]
    
    def _get_cache_key(self, features: np.ndarray) -> str:
        """Generate cache key from features."""
        feature_hash = self.feature_extractor.compute_feature_hash(features)
        return f"prediction:{feature_hash}"
    
    def _get_from_cache(self, key: str) -> Optional[Dict]:
        """Get result from cache."""
        if self.redis_client is None:
            return None
        
        try:
            import json
            cached = self.redis_client.get(key)
            if cached:
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
        
        return None
    
    def _set_cache(self, key: str, value: Dict):
        """Set result in cache."""
        if self.redis_client is None:
            return
        
        try:
            import json
            self.redis_client.setex(
                key,
                self.cache_ttl,
                json.dumps(value),
            )
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self._model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "version": self._model_version,
            "device": self._device,
            "path": self.model_path,
            "num_classes": self._model.num_classes,
            "parameters": sum(p.numel() for p in self._model.parameters()),
        }
    
    def warmup(self, num_iterations: int = 3):
        """
        Warm up the model with dummy predictions.
        
        This helps ensure consistent latency after startup.
        
        Args:
            num_iterations: Number of warmup iterations
        """
        logger.info("Warming up model...")
        
        dummy_features = np.random.randn(50, 7).astype(np.float32)
        
        for i in range(num_iterations):
            _ = self.predict(dummy_features)
        
        logger.info(f"Model warmup complete ({num_iterations} iterations)")


class PredictorPool:
    """
    Pool of predictor instances for concurrent inference.
    
    Useful for handling multiple concurrent requests with
    separate model instances.
    """
    
    def __init__(
        self,
        pool_size: int = 4,
        **predictor_kwargs,
    ):
        """
        Initialize predictor pool.
        
        Args:
            pool_size: Number of predictors in pool
            **predictor_kwargs: Arguments passed to Predictor
        """
        import threading
        from queue import Queue
        
        self.pool_size = pool_size
        self.predictors: Queue = Queue()
        self.lock = threading.Lock()
        
        # Create predictors
        for i in range(pool_size):
            # Create new instance (bypass singleton)
            predictor = object.__new__(Predictor)
            predictor._initialized = False
            predictor.__init__(**predictor_kwargs)
            self.predictors.put(predictor)
    
    def predict(self, features: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Make prediction using available predictor from pool.
        
        Args:
            features: Feature array
            **kwargs: Additional arguments for predict
        
        Returns:
            Prediction result
        """
        predictor = self.predictors.get()
        
        try:
            result = predictor.predict(features, **kwargs)
            return result
        finally:
            self.predictors.put(predictor)
