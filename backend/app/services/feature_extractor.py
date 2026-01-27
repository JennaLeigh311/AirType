"""
AirType Feature Extractor

This module provides kinematic feature extraction from stroke point sequences
for use with the handwriting recognition model.
"""

from typing import List, Dict, Any, Optional, Tuple
import hashlib

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d


class FeatureExtractor:
    """
    Extract kinematic features from stroke point sequences.
    
    Features extracted:
    - Position (x, y)
    - Velocity (vx, vy)
    - Acceleration magnitude
    - Curvature
    - Pen state (down/up)
    """
    
    def __init__(
        self,
        sequence_length: int = 50,
        smoothing_sigma: float = 1.0,
        normalize: bool = True,
    ):
        """
        Initialize the feature extractor.
        
        Args:
            sequence_length: Target sequence length for resampling
            smoothing_sigma: Gaussian smoothing sigma for velocity/acceleration
            normalize: Whether to normalize features
        """
        self.sequence_length = sequence_length
        self.smoothing_sigma = smoothing_sigma
        self.normalize = normalize
        
        # Pre-computed normalization statistics
        # These should be computed from training data
        self.feature_mean = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.feature_std = np.array([0.3, 0.3, 0.1, 0.1, 0.5, 1.0, 0.1])
    
    def extract_sequence_features(
        self,
        points: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Extract features from a sequence of points.
        
        Args:
            points: List of point dictionaries with x, y, timestamp_ms
        
        Returns:
            Feature array of shape (sequence_length, 7)
        """
        if len(points) < 2:
            # Return zero features for too few points
            return np.zeros((self.sequence_length, 7))
        
        # Convert to numpy array
        positions = np.array([
            [p.get("x", 0), p.get("y", 0), p.get("timestamp_ms", 0)]
            for p in points
        ])
        
        # Normalize stroke (center and scale)
        positions[:, :2] = self._normalize_stroke(positions[:, :2])
        
        # Compute kinematic features
        features = self._compute_kinematics(positions)
        
        # Resample to fixed length
        features = self._resample(features, self.sequence_length)
        
        # Apply normalization if enabled
        if self.normalize:
            features = self._apply_normalization(features)
        
        return features
    
    def calculate_point_features(
        self,
        prev_prev_point: Dict[str, Any],
        prev_point: Dict[str, Any],
        current_point: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Calculate features for a single point given previous points.
        
        Args:
            prev_prev_point: Point at t-2
            prev_point: Point at t-1
            current_point: Current point at t
        
        Returns:
            Dictionary with velocity, acceleration, curvature
        """
        # Extract coordinates
        x0, y0, t0 = prev_prev_point["x"], prev_prev_point["y"], prev_prev_point.get("timestamp", 0)
        x1, y1, t1 = prev_point["x"], prev_point["y"], prev_point.get("timestamp", 0)
        x2, y2, t2 = current_point["x"], current_point["y"], current_point.get("timestamp", 0)
        
        # Time differences (avoid division by zero)
        dt1 = max(t1 - t0, 1) / 1000.0  # Convert to seconds
        dt2 = max(t2 - t1, 1) / 1000.0
        
        # Velocities
        vx1 = (x1 - x0) / dt1
        vy1 = (y1 - y0) / dt1
        vx2 = (x2 - x1) / dt2
        vy2 = (y2 - y1) / dt2
        
        # Current velocity
        velocity_x = vx2
        velocity_y = vy2
        
        # Acceleration
        ax = (vx2 - vx1) / dt2
        ay = (vy2 - vy1) / dt2
        acceleration = np.sqrt(ax**2 + ay**2)
        
        # Curvature: κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        # Using numerical approximation
        x_prime = (vx1 + vx2) / 2
        y_prime = (vy1 + vy2) / 2
        x_double_prime = ax
        y_double_prime = ay
        
        numerator = abs(x_prime * y_double_prime - y_prime * x_double_prime)
        denominator = (x_prime**2 + y_prime**2)**(3/2)
        
        if denominator > 1e-8:
            curvature = numerator / denominator
        else:
            curvature = 0.0
        
        return {
            "velocity_x": float(velocity_x),
            "velocity_y": float(velocity_y),
            "acceleration": float(acceleration),
            "curvature": float(curvature),
        }
    
    def _normalize_stroke(self, positions: np.ndarray) -> np.ndarray:
        """
        Normalize stroke to center and unit scale.
        
        Args:
            positions: Array of shape (n, 2) with x, y coordinates
        
        Returns:
            Normalized positions
        """
        # Center at origin
        centroid = positions.mean(axis=0)
        centered = positions - centroid
        
        # Scale to unit square while preserving aspect ratio
        max_range = max(
            centered[:, 0].max() - centered[:, 0].min(),
            centered[:, 1].max() - centered[:, 1].min(),
        )
        
        if max_range > 1e-8:
            scaled = centered / max_range
        else:
            scaled = centered
        
        # Shift back to [0, 1] range
        normalized = scaled + 0.5
        
        return np.clip(normalized, 0, 1)
    
    def _compute_kinematics(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute kinematic features from positions.
        
        Args:
            positions: Array of shape (n, 3) with x, y, timestamp
        
        Returns:
            Feature array of shape (n, 7)
        """
        n = len(positions)
        features = np.zeros((n, 7))
        
        # Position features
        features[:, 0] = positions[:, 0]  # x
        features[:, 1] = positions[:, 1]  # y
        
        if n < 2:
            features[:, 6] = 1  # pen_state
            return features
        
        # Time differences (in seconds)
        dt = np.diff(positions[:, 2]) / 1000.0
        dt = np.maximum(dt, 1e-6)  # Avoid division by zero
        
        # Velocity (central difference)
        dx = np.diff(positions[:, 0])
        dy = np.diff(positions[:, 1])
        
        vx = dx / dt
        vy = dy / dt
        
        # Apply Gaussian smoothing
        if self.smoothing_sigma > 0 and len(vx) > 3:
            vx = gaussian_filter1d(vx, self.smoothing_sigma)
            vy = gaussian_filter1d(vy, self.smoothing_sigma)
        
        # Pad velocity to match length
        features[:-1, 2] = vx
        features[-1, 2] = vx[-1] if len(vx) > 0 else 0
        features[:-1, 3] = vy
        features[-1, 3] = vy[-1] if len(vy) > 0 else 0
        
        # Acceleration
        if n >= 3:
            dvx = np.diff(vx)
            dvy = np.diff(vy)
            dt2 = dt[:-1]
            
            ax = dvx / np.maximum(dt2, 1e-6)
            ay = dvy / np.maximum(dt2, 1e-6)
            
            # Apply smoothing
            if self.smoothing_sigma > 0 and len(ax) > 3:
                ax = gaussian_filter1d(ax, self.smoothing_sigma)
                ay = gaussian_filter1d(ay, self.smoothing_sigma)
            
            acc_magnitude = np.sqrt(ax**2 + ay**2)
            
            features[:-2, 4] = acc_magnitude
            features[-2, 4] = acc_magnitude[-1] if len(acc_magnitude) > 0 else 0
            features[-1, 4] = features[-2, 4]
        
        # Curvature
        if n >= 3:
            curvature = self._compute_curvature(
                features[:, 2], features[:, 3],
                features[:, 4]
            )
            features[:, 5] = curvature
        
        # Pen state (always down for single stroke)
        features[:, 6] = 1
        
        return features
    
    def _compute_curvature(
        self,
        vx: np.ndarray,
        vy: np.ndarray,
        acc: np.ndarray,
    ) -> np.ndarray:
        """
        Compute curvature from velocity and acceleration.
        
        κ = |v × a| / |v|^3
        """
        n = len(vx)
        curvature = np.zeros(n)
        
        for i in range(n):
            v_mag = np.sqrt(vx[i]**2 + vy[i]**2)
            
            if v_mag > 1e-8:
                # Approximate cross product in 2D (gives scalar)
                # For 2D, curvature = |vx*ay - vy*ax| / |v|^3
                # We approximate using acceleration magnitude
                curvature[i] = acc[i] / (v_mag**2 + 1e-8)
            else:
                curvature[i] = 0
        
        return np.clip(curvature, 0, 10)  # Clip extreme values
    
    def _resample(self, features: np.ndarray, target_length: int) -> np.ndarray:
        """
        Resample features to fixed length using linear interpolation.
        
        Args:
            features: Feature array of shape (n, d)
            target_length: Target sequence length
        
        Returns:
            Resampled array of shape (target_length, d)
        """
        current_length = len(features)
        
        if current_length == target_length:
            return features
        
        if current_length < 2:
            # Repeat single point
            return np.tile(features, (target_length, 1))[:target_length]
        
        # Create interpolation function for each feature
        x_old = np.linspace(0, 1, current_length)
        x_new = np.linspace(0, 1, target_length)
        
        resampled = np.zeros((target_length, features.shape[1]))
        
        for i in range(features.shape[1]):
            f = interp1d(x_old, features[:, i], kind="linear", fill_value="extrapolate")
            resampled[:, i] = f(x_new)
        
        return resampled
    
    def _apply_normalization(self, features: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to features.
        
        Args:
            features: Feature array
        
        Returns:
            Normalized features
        """
        # Avoid division by zero
        std = np.maximum(self.feature_std, 1e-8)
        
        normalized = (features - self.feature_mean) / std
        
        return normalized
    
    def compute_feature_hash(self, features: np.ndarray) -> str:
        """
        Compute SHA256 hash of feature array for deduplication.
        
        Args:
            features: Feature array
        
        Returns:
            Hex string of SHA256 hash
        """
        # Quantize to reduce noise sensitivity
        quantized = np.round(features, decimals=3)
        
        # Convert to bytes
        feature_bytes = quantized.tobytes()
        
        # Compute hash
        return hashlib.sha256(feature_bytes).hexdigest()
    
    def compute_similarity(
        self,
        features1: np.ndarray,
        features2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two feature sequences.
        
        Args:
            features1: First feature array
            features2: Second feature array
        
        Returns:
            Cosine similarity score (0 to 1)
        """
        # Flatten features
        flat1 = features1.flatten()
        flat2 = features2.flatten()
        
        # Compute cosine similarity
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        
        similarity = np.dot(flat1, flat2) / (norm1 * norm2)
        
        return float(max(0, similarity))
    
    def set_normalization_stats(
        self,
        mean: np.ndarray,
        std: np.ndarray,
    ):
        """
        Set normalization statistics from training data.
        
        Args:
            mean: Feature means
            std: Feature standard deviations
        """
        self.feature_mean = mean
        self.feature_std = std


class SequenceWindower:
    """
    Handles windowing of multiple strokes for context-aware prediction.
    """
    
    def __init__(
        self,
        context_window: int = 3,
        max_sequence_length: int = 150,
    ):
        """
        Initialize the windower.
        
        Args:
            context_window: Number of previous strokes to include
            max_sequence_length: Maximum combined sequence length
        """
        self.context_window = context_window
        self.max_sequence_length = max_sequence_length
        self.stroke_buffer: List[np.ndarray] = []
    
    def add_stroke(self, features: np.ndarray):
        """
        Add a stroke to the buffer.
        
        Args:
            features: Feature array for the stroke
        """
        self.stroke_buffer.append(features)
        
        # Keep only recent strokes
        if len(self.stroke_buffer) > self.context_window:
            self.stroke_buffer.pop(0)
    
    def get_context_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get combined features from recent strokes with mask.
        
        Returns:
            Tuple of (features, mask)
            - features: Combined feature array
            - mask: Boolean mask (True for valid positions)
        """
        if not self.stroke_buffer:
            features = np.zeros((self.max_sequence_length, 7))
            mask = np.zeros(self.max_sequence_length, dtype=bool)
            return features, mask
        
        # Concatenate stroke features
        combined = np.vstack(self.stroke_buffer)
        
        # Truncate if too long
        if len(combined) > self.max_sequence_length:
            combined = combined[-self.max_sequence_length:]
        
        # Create output array with padding
        features = np.zeros((self.max_sequence_length, combined.shape[1]))
        features[:len(combined)] = combined
        
        # Create mask
        mask = np.zeros(self.max_sequence_length, dtype=bool)
        mask[:len(combined)] = True
        
        return features, mask
    
    def clear(self):
        """Clear the stroke buffer."""
        self.stroke_buffer = []
