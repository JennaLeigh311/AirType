"""
AirType Stroke Dataset

This module provides PyTorch Dataset and DataLoader implementations
for loading and preprocessing stroke data for model training.
"""

from typing import List, Dict, Tuple, Optional, Callable
import random
import math

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class StrokeDataset(Dataset):
    """
    PyTorch Dataset for stroke sequences.
    
    Handles loading, preprocessing, and augmentation of stroke data
    for training the handwriting recognition model.
    """
    
    def __init__(
        self,
        strokes: List[Dict],
        labels: List[str],
        sequence_length: int = 50,
        augment: bool = True,
        normalize: bool = True,
        feature_stats: Optional[Dict] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            strokes: List of stroke dictionaries, each containing point lists
            labels: List of character labels
            sequence_length: Fixed sequence length for resampling
            augment: Whether to apply data augmentation
            normalize: Whether to normalize features
            feature_stats: Pre-computed feature statistics for normalization
        """
        self.strokes = strokes
        self.labels = labels
        self.sequence_length = sequence_length
        self.augment = augment
        self.normalize = normalize
        self.feature_stats = feature_stats
        
        # Character mappings
        self.chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        self.char_to_idx = {c: i for i, c in enumerate(self.chars)}
        
        # Compute feature statistics if not provided
        if self.normalize and self.feature_stats is None:
            self.feature_stats = self._compute_feature_stats()
    
    def __len__(self) -> int:
        return len(self.strokes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (features, label_idx, sequence_length)
        """
        stroke = self.strokes[idx]
        label = self.labels[idx]
        
        # Extract points from stroke
        points = stroke.get("points", stroke)
        if isinstance(points, dict):
            points = [points]
        
        # Convert to numpy array
        points_array = self._points_to_array(points)
        
        # Apply augmentation if enabled
        if self.augment:
            points_array = self._augment(points_array)
        
        # Compute kinematic features
        features = self._compute_features(points_array)
        
        # Resample to fixed length
        features = self._resample(features, self.sequence_length)
        
        # Normalize
        if self.normalize:
            features = self._normalize_features(features)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features)
        
        # Get label index
        label_idx = self.char_to_idx.get(label, 0)
        
        return features_tensor, label_idx, len(features)
    
    def _points_to_array(self, points: List[Dict]) -> np.ndarray:
        """Convert list of point dictionaries to numpy array."""
        arr = np.array([
            [p.get("x", 0), p.get("y", 0), p.get("timestamp_ms", 0)]
            for p in points
        ])
        return arr
    
    def _compute_features(self, points: np.ndarray) -> np.ndarray:
        """
        Compute kinematic features from raw points.
        
        Features: x, y, velocity_x, velocity_y, acceleration, curvature, pen_state
        """
        n_points = len(points)
        features = np.zeros((n_points, 7))
        
        # Position features
        features[:, 0] = points[:, 0]  # x
        features[:, 1] = points[:, 1]  # y
        
        if n_points < 3:
            # Not enough points for derivatives
            features[:, 6] = 1  # pen_state = down
            return features
        
        # Time differences
        dt = np.diff(points[:, 2])
        dt = np.maximum(dt, 1)  # Avoid division by zero
        
        # Velocity (central difference)
        dx = np.diff(points[:, 0])
        dy = np.diff(points[:, 1])
        
        vx = dx / dt
        vy = dy / dt
        
        # Pad velocity to match length
        features[1:, 2] = vx  # velocity_x
        features[1:, 3] = vy  # velocity_y
        features[0, 2] = vx[0] if len(vx) > 0 else 0
        features[0, 3] = vy[0] if len(vy) > 0 else 0
        
        # Acceleration
        if n_points >= 3:
            dvx = np.diff(vx)
            dvy = np.diff(vy)
            dt2 = dt[:-1]
            
            ax = dvx / np.maximum(dt2, 1)
            ay = dvy / np.maximum(dt2, 1)
            
            acc = np.sqrt(ax**2 + ay**2)
            features[2:, 4] = acc
            features[1, 4] = acc[0] if len(acc) > 0 else 0
        
        # Curvature
        if n_points >= 3:
            # κ = |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
            x_prime = vx
            y_prime = vy
            
            if len(ax) > 0 and len(ay) > 0:
                x_double_prime = np.zeros_like(x_prime)
                y_double_prime = np.zeros_like(y_prime)
                x_double_prime[:-1] = ax
                y_double_prime[:-1] = ay
                
                numerator = np.abs(x_prime * y_double_prime - y_prime * x_double_prime)
                denominator = (x_prime**2 + y_prime**2)**(3/2)
                denominator = np.maximum(denominator, 1e-8)
                
                curvature = numerator / denominator
                features[1:, 5] = curvature
        
        # Pen state (always down for now)
        features[:, 6] = 1
        
        return features
    
    def _augment(self, points: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation.
        
        Augmentations:
        - Temporal jitter: ±5% timestamp noise
        - Spatial rotation: ±15 degrees
        - Scaling: 0.9-1.1x
        - Translation jitter
        """
        augmented = points.copy()
        
        # Temporal jitter
        if random.random() < 0.5:
            jitter = np.random.uniform(-0.05, 0.05, len(points))
            time_range = points[-1, 2] - points[0, 2]
            augmented[:, 2] += jitter * time_range
        
        # Spatial rotation
        if random.random() < 0.5:
            angle = np.random.uniform(-15, 15) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            
            # Center of mass
            cx = augmented[:, 0].mean()
            cy = augmented[:, 1].mean()
            
            # Rotate around center
            x_centered = augmented[:, 0] - cx
            y_centered = augmented[:, 1] - cy
            
            augmented[:, 0] = x_centered * cos_a - y_centered * sin_a + cx
            augmented[:, 1] = x_centered * sin_a + y_centered * cos_a + cy
        
        # Scaling
        if random.random() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            cx = augmented[:, 0].mean()
            cy = augmented[:, 1].mean()
            
            augmented[:, 0] = (augmented[:, 0] - cx) * scale + cx
            augmented[:, 1] = (augmented[:, 1] - cy) * scale + cy
        
        # Translation jitter
        if random.random() < 0.3:
            tx = np.random.uniform(-0.05, 0.05)
            ty = np.random.uniform(-0.05, 0.05)
            augmented[:, 0] += tx
            augmented[:, 1] += ty
        
        # Clip to valid range
        augmented[:, 0] = np.clip(augmented[:, 0], 0, 1)
        augmented[:, 1] = np.clip(augmented[:, 1], 0, 1)
        
        return augmented
    
    def _resample(self, features: np.ndarray, target_length: int) -> np.ndarray:
        """
        Resample features to fixed length using linear interpolation.
        """
        current_length = len(features)
        
        if current_length == target_length:
            return features
        
        # Interpolation indices
        indices = np.linspace(0, current_length - 1, target_length)
        
        # Interpolate each feature dimension
        resampled = np.zeros((target_length, features.shape[1]))
        for i in range(features.shape[1]):
            resampled[:, i] = np.interp(indices, np.arange(current_length), features[:, i])
        
        return resampled
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using pre-computed statistics."""
        if self.feature_stats is None:
            return features
        
        mean = self.feature_stats.get("mean", np.zeros(features.shape[1]))
        std = self.feature_stats.get("std", np.ones(features.shape[1]))
        
        # Avoid division by zero
        std = np.maximum(std, 1e-8)
        
        normalized = (features - mean) / std
        return normalized
    
    def _compute_feature_stats(self) -> Dict[str, np.ndarray]:
        """Compute mean and std for all features across the dataset."""
        all_features = []
        
        for stroke in self.strokes[:min(1000, len(self.strokes))]:  # Sample up to 1000
            points = stroke.get("points", stroke)
            if isinstance(points, dict):
                points = [points]
            
            points_array = self._points_to_array(points)
            features = self._compute_features(points_array)
            all_features.append(features)
        
        if not all_features:
            return {"mean": np.zeros(7), "std": np.ones(7)}
        
        all_features = np.vstack(all_features)
        
        return {
            "mean": np.mean(all_features, axis=0),
            "std": np.std(all_features, axis=0),
        }
    
    def get_feature_stats(self) -> Dict[str, np.ndarray]:
        """Get the computed feature statistics."""
        return self.feature_stats


def collate_fn(
    batch: List[Tuple[torch.Tensor, int, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.
    
    Args:
        batch: List of (features, label, length) tuples
    
    Returns:
        Tuple of (padded_features, labels, lengths, mask)
    """
    features, labels, lengths = zip(*batch)
    
    # Pad sequences
    padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    
    # Create mask
    max_len = padded.shape[1]
    mask = torch.zeros(len(batch), max_len)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    
    labels = torch.LongTensor(labels)
    lengths = torch.LongTensor(lengths)
    
    return padded, labels, lengths, mask


def create_data_loaders(
    train_strokes: List[Dict],
    train_labels: List[str],
    val_strokes: Optional[List[Dict]] = None,
    val_labels: Optional[List[str]] = None,
    batch_size: int = 32,
    sequence_length: int = 50,
    num_workers: int = 4,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation data loaders.
    
    Args:
        train_strokes: Training stroke data
        train_labels: Training labels
        val_strokes: Validation stroke data
        val_labels: Validation labels
        batch_size: Batch size
        sequence_length: Fixed sequence length
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create training dataset
    train_dataset = StrokeDataset(
        strokes=train_strokes,
        labels=train_labels,
        sequence_length=sequence_length,
        augment=True,
        normalize=True,
    )
    
    # Get feature stats for validation
    feature_stats = train_dataset.get_feature_stats()
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    val_loader = None
    if val_strokes is not None and val_labels is not None:
        val_dataset = StrokeDataset(
            strokes=val_strokes,
            labels=val_labels,
            sequence_length=sequence_length,
            augment=False,  # No augmentation for validation
            normalize=True,
            feature_stats=feature_stats,  # Use training stats
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return train_loader, val_loader


class SyntheticStrokeGenerator:
    """
    Generate synthetic stroke data for training.
    
    Creates synthetic handwriting strokes by combining basic shapes
    and patterns characteristic of each character.
    """
    
    def __init__(self, num_points: int = 50):
        """
        Initialize the generator.
        
        Args:
            num_points: Number of points per stroke
        """
        self.num_points = num_points
        self.chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    def generate_batch(
        self,
        batch_size: int,
        chars: Optional[str] = None,
    ) -> Tuple[List[Dict], List[str]]:
        """
        Generate a batch of synthetic strokes.
        
        Args:
            batch_size: Number of strokes to generate
            chars: Characters to generate (defaults to all)
        
        Returns:
            Tuple of (strokes, labels)
        """
        if chars is None:
            chars = self.chars
        
        strokes = []
        labels = []
        
        for _ in range(batch_size):
            char = random.choice(chars)
            stroke = self._generate_char_stroke(char)
            strokes.append({"points": stroke})
            labels.append(char)
        
        return strokes, labels
    
    def _generate_char_stroke(self, char: str) -> List[Dict]:
        """Generate a synthetic stroke for a character."""
        t = np.linspace(0, 1, self.num_points)
        
        # Base shape based on character type
        if char.isdigit():
            points = self._generate_digit_shape(char, t)
        elif char.isupper():
            points = self._generate_uppercase_shape(char, t)
        else:
            points = self._generate_lowercase_shape(char, t)
        
        # Add noise for variation
        noise = np.random.normal(0, 0.02, points.shape)
        points += noise
        
        # Normalize to [0, 1]
        points[:, 0] = (points[:, 0] - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min() + 1e-8)
        points[:, 1] = (points[:, 1] - points[:, 1].min()) / (points[:, 1].max() - points[:, 1].min() + 1e-8)
        
        # Scale and center
        points = points * 0.8 + 0.1
        
        # Convert to point list
        return [
            {
                "x": float(points[i, 0]),
                "y": float(points[i, 1]),
                "timestamp_ms": int(i * 10),
            }
            for i in range(len(points))
        ]
    
    def _generate_digit_shape(self, digit: str, t: np.ndarray) -> np.ndarray:
        """Generate shape for a digit."""
        n = len(t)
        points = np.zeros((n, 2))
        
        if digit == "0":
            # Ellipse
            points[:, 0] = np.cos(2 * np.pi * t)
            points[:, 1] = np.sin(2 * np.pi * t) * 1.5
        elif digit == "1":
            # Vertical line with small hook
            points[:, 0] = 0.1 * np.sin(np.pi * t)
            points[:, 1] = t * 2
        elif digit == "2":
            # Curved top + diagonal + base
            points[:n//3, 0] = np.cos(np.pi * (1 - t[:n//3] * 3))
            points[:n//3, 1] = np.sin(np.pi * (1 - t[:n//3] * 3)) + 1
            points[n//3:2*n//3, 0] = t[n//3:2*n//3] * 3 - 1
            points[n//3:2*n//3, 1] = 2 - t[n//3:2*n//3] * 3 * 2
            points[2*n//3:, 0] = t[2*n//3:] * 3 - 1
            points[2*n//3:, 1] = 0
        else:
            # Generic curve for other digits
            points[:, 0] = np.cos(2 * np.pi * t + int(digit) * 0.5)
            points[:, 1] = np.sin(2 * np.pi * t + int(digit) * 0.3) + t
        
        return points
    
    def _generate_uppercase_shape(self, char: str, t: np.ndarray) -> np.ndarray:
        """Generate shape for an uppercase letter."""
        n = len(t)
        points = np.zeros((n, 2))
        
        # Character index for variation
        idx = ord(char) - ord("A")
        
        # Generic uppercase shapes with variation
        if char in "OQCG":
            # Round letters
            points[:, 0] = np.cos(2 * np.pi * t * (1 + idx * 0.01))
            points[:, 1] = np.sin(2 * np.pi * t) * 1.5
        elif char in "ILHT":
            # Straight letters
            points[:n//2, 0] = 0
            points[:n//2, 1] = t[:n//2] * 2 * 2
            points[n//2:, 0] = t[n//2:] * 2 - 1
            points[n//2:, 1] = 1
        else:
            # Mixed letters
            points[:, 0] = np.sin(np.pi * t) + idx * 0.05
            points[:, 1] = t * 2
        
        return points
    
    def _generate_lowercase_shape(self, char: str, t: np.ndarray) -> np.ndarray:
        """Generate shape for a lowercase letter."""
        n = len(t)
        points = np.zeros((n, 2))
        
        # Character index for variation
        idx = ord(char) - ord("a")
        
        # Generic lowercase shapes with variation
        if char in "ocea":
            # Round letters
            points[:, 0] = np.cos(2 * np.pi * t) * 0.8
            points[:, 1] = np.sin(2 * np.pi * t)
        elif char in "ilj":
            # Straight letters
            points[:, 0] = 0.1 * np.sin(np.pi * t)
            points[:, 1] = t * 1.5
        else:
            # Mixed letters
            points[:, 0] = np.sin(np.pi * t * (1 + idx * 0.02)) * 0.8
            points[:, 1] = t * 1.2 + np.cos(np.pi * t) * 0.2
        
        return points
