"""
AirType Data Augmentation Module

Provides comprehensive data augmentation for handwriting stroke data
to artificially increase training data and improve model generalization.
"""

import numpy as np
import random
from typing import List, Dict, Tuple, Optional


class StrokeAugmentor:
    """
    Augmentation pipeline for handwriting stroke data.
    
    Applies various transformations to increase training data variety
    without losing the essential characteristics of each character.
    """
    
    def __init__(
        self,
        rotation_range: float = 15.0,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        translation_range: float = 0.1,
        noise_level: float = 0.02,
        temporal_jitter: float = 0.05,
        shear_range: float = 0.1,
        aspect_ratio_range: Tuple[float, float] = (0.9, 1.1),
    ):
        """
        Initialize augmentor with transformation parameters.
        
        Args:
            rotation_range: Max rotation angle in degrees (±)
            scale_range: Scale factor range (min, max)
            translation_range: Translation factor as fraction of stroke size
            noise_level: Gaussian noise standard deviation
            temporal_jitter: Timestamp noise factor (±)
            shear_range: Shear transformation range
            aspect_ratio_range: Aspect ratio modification range
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.noise_level = noise_level
        self.temporal_jitter = temporal_jitter
        self.shear_range = shear_range
        self.aspect_ratio_range = aspect_ratio_range
    
    def augment(
        self,
        features: List[List[float]],
        augmentation_factor: int = 5,
        include_original: bool = True
    ) -> List[List[List[float]]]:
        """
        Generate augmented versions of a single stroke sample.
        
        Args:
            features: Original stroke features (2D array)
            augmentation_factor: Number of augmented versions to create
            include_original: Whether to include original in output
        
        Returns:
            List of augmented feature arrays
        """
        augmented = []
        
        if include_original:
            augmented.append(features)
        
        features_np = np.array(features)
        
        for _ in range(augmentation_factor):
            aug_features = self._apply_random_augmentations(features_np)
            augmented.append(aug_features.tolist())
        
        return augmented
    
    def _apply_random_augmentations(self, features: np.ndarray) -> np.ndarray:
        """Apply random combination of augmentations."""
        result = features.copy()
        
        # Extract x, y coordinates (assuming first two columns)
        x = result[:, 0]
        y = result[:, 1]
        
        # Center the stroke
        cx, cy = np.mean(x), np.mean(y)
        x_centered = x - cx
        y_centered = y - cy
        
        # Random rotation
        if random.random() < 0.8:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            x_centered, y_centered = self._rotate(x_centered, y_centered, angle)
        
        # Random scaling
        if random.random() < 0.8:
            scale = random.uniform(*self.scale_range)
            x_centered *= scale
            y_centered *= scale
        
        # Random shear
        if random.random() < 0.5:
            shear = random.uniform(-self.shear_range, self.shear_range)
            x_centered = x_centered + shear * y_centered
        
        # Random aspect ratio change
        if random.random() < 0.5:
            aspect = random.uniform(*self.aspect_ratio_range)
            x_centered *= aspect
        
        # Random translation
        if random.random() < 0.7:
            stroke_width = np.max(x_centered) - np.min(x_centered)
            stroke_height = np.max(y_centered) - np.min(y_centered)
            tx = random.uniform(-self.translation_range, self.translation_range) * stroke_width
            ty = random.uniform(-self.translation_range, self.translation_range) * stroke_height
            x_centered += tx
            y_centered += ty
        
        # Add Gaussian noise to positions
        if random.random() < 0.6:
            stroke_scale = max(np.std(x_centered), np.std(y_centered), 1)
            noise_x = np.random.normal(0, self.noise_level * stroke_scale, len(x_centered))
            noise_y = np.random.normal(0, self.noise_level * stroke_scale, len(y_centered))
            x_centered += noise_x
            y_centered += noise_y
        
        # Restore to original center
        result[:, 0] = x_centered + cx
        result[:, 1] = y_centered + cy
        
        # Temporal jitter (if timestamp column exists)
        if result.shape[1] > 2:
            if random.random() < 0.5:
                timestamps = result[:, 2]
                jitter = np.random.uniform(
                    1 - self.temporal_jitter,
                    1 + self.temporal_jitter,
                    len(timestamps)
                )
                # Apply cumulative jitter to maintain ordering
                time_diffs = np.diff(timestamps, prepend=timestamps[0])
                time_diffs = time_diffs * jitter
                result[:, 2] = np.cumsum(time_diffs)
        
        # Add noise to velocity features if present (columns 2-3 or 3-4)
        if result.shape[1] >= 5:
            if random.random() < 0.4:
                velocity_noise = np.random.normal(0, 0.05, (len(result), 2))
                result[:, 2:4] += velocity_noise
        
        return result
    
    def _rotate(
        self,
        x: np.ndarray,
        y: np.ndarray,
        angle: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Rotate points by angle (in degrees)."""
        theta = np.radians(angle)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_rot = x * cos_t - y * sin_t
        y_rot = x * sin_t + y * cos_t
        return x_rot, y_rot


class BatchAugmentor:
    """
    Batch augmentation for training datasets.
    
    Efficiently augments entire datasets with class balancing.
    """
    
    def __init__(self, augmentor: Optional[StrokeAugmentor] = None):
        """
        Initialize batch augmentor.
        
        Args:
            augmentor: StrokeAugmentor instance (creates default if None)
        """
        self.augmentor = augmentor or StrokeAugmentor()
    
    def augment_dataset(
        self,
        features: List[List[List[float]]],
        labels: List[str],
        target_samples_per_class: int = 100,
        max_augmentation_factor: int = 10
    ) -> Tuple[List[List[List[float]]], List[str]]:
        """
        Augment dataset with class balancing.
        
        Generates more augmented samples for underrepresented classes.
        
        Args:
            features: List of stroke feature arrays
            labels: List of character labels
            target_samples_per_class: Desired samples per class
            max_augmentation_factor: Maximum augmentation multiplier
        
        Returns:
            Tuple of (augmented_features, augmented_labels)
        """
        # Count samples per class
        class_counts = {}
        class_samples = {}
        
        for feat, label in zip(features, labels):
            if label not in class_counts:
                class_counts[label] = 0
                class_samples[label] = []
            class_counts[label] += 1
            class_samples[label].append(feat)
        
        augmented_features = []
        augmented_labels = []
        
        for label, samples in class_samples.items():
            current_count = len(samples)
            
            # Calculate augmentation factor to reach target
            if current_count >= target_samples_per_class:
                aug_factor = 1  # Just include originals
            else:
                needed = target_samples_per_class - current_count
                aug_factor = min(
                    max(1, needed // current_count + 1),
                    max_augmentation_factor
                )
            
            # Augment each sample
            for sample in samples:
                if aug_factor > 1:
                    aug_samples = self.augmentor.augment(
                        sample,
                        augmentation_factor=aug_factor - 1,
                        include_original=True
                    )
                else:
                    aug_samples = [sample]
                
                for aug_sample in aug_samples:
                    augmented_features.append(aug_sample)
                    augmented_labels.append(label)
        
        return augmented_features, augmented_labels
    
    def augment_batch(
        self,
        features: List[List[List[float]]],
        labels: List[str],
        augmentation_factor: int = 5
    ) -> Tuple[List[List[List[float]]], List[str]]:
        """
        Simple batch augmentation without class balancing.
        
        Args:
            features: List of stroke feature arrays
            labels: List of character labels
            augmentation_factor: Number of augmented versions per sample
        
        Returns:
            Tuple of (augmented_features, augmented_labels)
        """
        augmented_features = []
        augmented_labels = []
        
        for feat, label in zip(features, labels):
            aug_samples = self.augmentor.augment(
                feat,
                augmentation_factor=augmentation_factor,
                include_original=True
            )
            for aug_sample in aug_samples:
                augmented_features.append(aug_sample)
                augmented_labels.append(label)
        
        return augmented_features, augmented_labels
