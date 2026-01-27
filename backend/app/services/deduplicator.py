"""
AirType Deduplicator Service

This module provides stroke deduplication to avoid redundant predictions
for similar or identical stroke patterns.
"""

from typing import Optional, Dict, List, Any
import logging
import time
import zlib
from datetime import datetime, timedelta

import numpy as np
import redis

from app.services.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class Deduplicator:
    """
    Handles stroke deduplication using feature hashing and similarity comparison.
    
    Features:
    - SHA256 feature hashing for exact match detection
    - Cosine similarity for near-duplicate detection
    - Redis caching for distributed deduplication
    - Configurable similarity threshold
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.95,
        redis_url: Optional[str] = None,
        cache_ttl: int = 3600,
        max_recent_strokes: int = 100,
    ):
        """
        Initialize the deduplicator.
        
        Args:
            similarity_threshold: Cosine similarity threshold for duplicates
            redis_url: Redis URL for distributed caching
            cache_ttl: Cache TTL in seconds
            max_recent_strokes: Maximum number of recent strokes to check
        """
        self.similarity_threshold = similarity_threshold
        self.cache_ttl = cache_ttl
        self.max_recent_strokes = max_recent_strokes
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(
            sequence_length=50,
            normalize=True,
        )
        
        # Redis client
        self.redis_client: Optional[redis.Redis] = None
        
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Deduplicator Redis connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using local cache.")
                self.redis_client = None
        
        # Local cache fallback
        self._local_cache: Dict[str, Dict] = {}
        self._local_features: Dict[str, np.ndarray] = {}
    
    def check_duplicate(
        self,
        features: np.ndarray,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Check if stroke features match a recent duplicate.
        
        Args:
            features: Feature array for the stroke
            user_id: User ID for scoping
        
        Returns:
            Cached prediction if duplicate found, None otherwise
        """
        # Compute feature hash
        feature_hash = self.feature_extractor.compute_feature_hash(features)
        
        # Check exact match first (fastest)
        exact_match = self._check_exact_match(feature_hash, user_id)
        if exact_match:
            logger.info(f"Exact duplicate found for user {user_id}")
            return exact_match
        
        # Check similarity match (slower but catches near-duplicates)
        similar_match = self._check_similar_match(features, user_id)
        if similar_match:
            logger.info(f"Similar duplicate found for user {user_id} (similarity: {similar_match.get('similarity', 'N/A')})")
            return similar_match
        
        return None
    
    def cache_features(
        self,
        session_id: str,
        features: np.ndarray,
        prediction_result: Dict[str, Any],
        user_id: Optional[str] = None,
    ):
        """
        Cache features and prediction for future deduplication.
        
        Args:
            session_id: Stroke session ID
            features: Feature array
            prediction_result: Prediction result to cache
            user_id: User ID for scoping
        """
        feature_hash = self.feature_extractor.compute_feature_hash(features)
        
        cache_data = {
            "session_id": str(session_id),
            "feature_hash": feature_hash,
            "predicted_char": prediction_result.get("predicted_char"),
            "confidence": prediction_result.get("confidence"),
            "alternatives": prediction_result.get("alternatives", []),
            "cached_at": datetime.utcnow().isoformat(),
        }
        
        # Compress features for storage
        compressed_features = zlib.compress(features.tobytes())
        
        if self.redis_client:
            try:
                # Store in Redis
                hash_key = f"stroke:hash:{user_id}:{feature_hash}"
                features_key = f"stroke:features:{user_id}:{session_id}"
                user_strokes_key = f"user:strokes:{user_id}"
                
                import json
                
                # Store hash -> prediction mapping
                self.redis_client.setex(
                    hash_key,
                    self.cache_ttl,
                    json.dumps(cache_data),
                )
                
                # Store compressed features
                self.redis_client.setex(
                    features_key,
                    self.cache_ttl,
                    compressed_features,
                )
                
                # Add to user's recent strokes list
                self.redis_client.lpush(user_strokes_key, session_id)
                self.redis_client.ltrim(user_strokes_key, 0, self.max_recent_strokes - 1)
                self.redis_client.expire(user_strokes_key, self.cache_ttl)
                
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")
                self._cache_local(feature_hash, features, cache_data, user_id)
        else:
            self._cache_local(feature_hash, features, cache_data, user_id)
    
    def _check_exact_match(
        self,
        feature_hash: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Check for exact hash match."""
        if self.redis_client:
            try:
                import json
                
                hash_key = f"stroke:hash:{user_id}:{feature_hash}"
                cached = self.redis_client.get(hash_key)
                
                if cached:
                    return json.loads(cached)
                    
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")
        
        # Check local cache
        local_key = f"{user_id}:{feature_hash}"
        if local_key in self._local_cache:
            cached = self._local_cache[local_key]
            
            # Check if not expired
            cached_at = datetime.fromisoformat(cached.get("cached_at", ""))
            if datetime.utcnow() - cached_at < timedelta(seconds=self.cache_ttl):
                return cached
            else:
                del self._local_cache[local_key]
        
        return None
    
    def _check_similar_match(
        self,
        features: np.ndarray,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Check for similar stroke using cosine similarity."""
        recent_strokes = self._get_recent_strokes(user_id)
        
        if not recent_strokes:
            return None
        
        for stroke_id in recent_strokes[:self.max_recent_strokes]:
            cached_features = self._get_cached_features(stroke_id, user_id)
            
            if cached_features is None:
                continue
            
            # Compute similarity
            similarity = self.feature_extractor.compute_similarity(
                features,
                cached_features,
            )
            
            if similarity >= self.similarity_threshold:
                # Get cached prediction
                cached_prediction = self._get_cached_prediction(stroke_id, user_id)
                
                if cached_prediction:
                    cached_prediction["similarity"] = similarity
                    cached_prediction["duplicate_of"] = stroke_id
                    return cached_prediction
        
        return None
    
    def _get_recent_strokes(self, user_id: str) -> List[str]:
        """Get list of recent stroke session IDs for user."""
        if self.redis_client:
            try:
                user_strokes_key = f"user:strokes:{user_id}"
                strokes = self.redis_client.lrange(user_strokes_key, 0, self.max_recent_strokes - 1)
                return [s.decode() if isinstance(s, bytes) else s for s in strokes]
            except Exception as e:
                logger.warning(f"Failed to get recent strokes: {e}")
        
        return []
    
    def _get_cached_features(
        self,
        session_id: str,
        user_id: str,
    ) -> Optional[np.ndarray]:
        """Get cached features for a session."""
        if self.redis_client:
            try:
                features_key = f"stroke:features:{user_id}:{session_id}"
                compressed = self.redis_client.get(features_key)
                
                if compressed:
                    decompressed = zlib.decompress(compressed)
                    features = np.frombuffer(decompressed, dtype=np.float64)
                    # Reshape to (seq_len, 7) - assuming 50 * 7 = 350 elements
                    return features.reshape(-1, 7)
                    
            except Exception as e:
                logger.warning(f"Failed to get cached features: {e}")
        
        # Check local cache
        local_key = f"{user_id}:{session_id}"
        return self._local_features.get(local_key)
    
    def _get_cached_prediction(
        self,
        session_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get cached prediction for a session."""
        # We need to look up by session_id, not hash
        # This requires either storing a mapping or iterating
        
        if self.redis_client:
            try:
                import json
                
                # Scan for matching session in cache
                pattern = f"stroke:hash:{user_id}:*"
                for key in self.redis_client.scan_iter(pattern, count=100):
                    cached = self.redis_client.get(key)
                    if cached:
                        data = json.loads(cached)
                        if data.get("session_id") == session_id:
                            return data
                            
            except Exception as e:
                logger.warning(f"Failed to get cached prediction: {e}")
        
        # Check local cache
        for key, data in self._local_cache.items():
            if key.startswith(f"{user_id}:") and data.get("session_id") == session_id:
                return data
        
        return None
    
    def _cache_local(
        self,
        feature_hash: str,
        features: np.ndarray,
        cache_data: Dict,
        user_id: str,
    ):
        """Store in local cache."""
        local_key = f"{user_id}:{feature_hash}"
        self._local_cache[local_key] = cache_data
        
        session_id = cache_data.get("session_id")
        if session_id:
            features_key = f"{user_id}:{session_id}"
            self._local_features[features_key] = features.copy()
        
        # Cleanup old entries
        self._cleanup_local_cache()
    
    def _cleanup_local_cache(self):
        """Remove expired entries from local cache."""
        now = datetime.utcnow()
        expired_keys = []
        
        for key, data in self._local_cache.items():
            try:
                cached_at = datetime.fromisoformat(data.get("cached_at", ""))
                if now - cached_at > timedelta(seconds=self.cache_ttl):
                    expired_keys.append(key)
            except (ValueError, TypeError):
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._local_cache[key]
        
        # Also cleanup features cache
        for key in list(self._local_features.keys()):
            if key not in self._local_cache:
                user_id = key.split(":")[0]
                for cache_key, data in self._local_cache.items():
                    if cache_key.startswith(f"{user_id}:"):
                        if data.get("session_id") == key.split(":")[1]:
                            break
                else:
                    del self._local_features[key]
    
    def clear_user_cache(self, user_id: str):
        """Clear all cached data for a user."""
        if self.redis_client:
            try:
                # Delete user's stroke list
                user_strokes_key = f"user:strokes:{user_id}"
                self.redis_client.delete(user_strokes_key)
                
                # Delete hash mappings
                for key in self.redis_client.scan_iter(f"stroke:hash:{user_id}:*"):
                    self.redis_client.delete(key)
                
                # Delete features
                for key in self.redis_client.scan_iter(f"stroke:features:{user_id}:*"):
                    self.redis_client.delete(key)
                    
            except Exception as e:
                logger.warning(f"Failed to clear user cache: {e}")
        
        # Clear local cache
        keys_to_delete = [k for k in self._local_cache.keys() if k.startswith(f"{user_id}:")]
        for key in keys_to_delete:
            del self._local_cache[key]
        
        keys_to_delete = [k for k in self._local_features.keys() if k.startswith(f"{user_id}:")]
        for key in keys_to_delete:
            del self._local_features[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        stats = {
            "local_cache_size": len(self._local_cache),
            "local_features_size": len(self._local_features),
            "similarity_threshold": self.similarity_threshold,
            "cache_ttl": self.cache_ttl,
        }
        
        if self.redis_client:
            try:
                # Count keys in Redis
                hash_count = sum(1 for _ in self.redis_client.scan_iter("stroke:hash:*", count=1000))
                features_count = sum(1 for _ in self.redis_client.scan_iter("stroke:features:*", count=1000))
                
                stats["redis_hash_count"] = hash_count
                stats["redis_features_count"] = features_count
                stats["redis_connected"] = True
            except Exception as e:
                stats["redis_connected"] = False
                stats["redis_error"] = str(e)
        else:
            stats["redis_connected"] = False
        
        return stats


def store_features_in_db(
    session_id: str,
    features: np.ndarray,
    feature_hash: str,
):
    """
    Store normalized features in database for permanent storage.
    
    Args:
        session_id: Stroke session UUID
        features: Normalized feature array
        feature_hash: SHA256 hash of features
    """
    from app.models import db, StrokeFeaturesCache
    
    # Compress features
    compressed = zlib.compress(features.tobytes())
    
    # Create or update cache entry
    cache_entry = StrokeFeaturesCache.query.get(session_id)
    
    if cache_entry:
        cache_entry.normalized_features = compressed
        cache_entry.feature_hash = feature_hash
    else:
        cache_entry = StrokeFeaturesCache(
            session_id=session_id,
            normalized_features=compressed,
            feature_hash=feature_hash,
        )
        db.session.add(cache_entry)
    
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error(f"Failed to store features in DB: {e}")
        raise
