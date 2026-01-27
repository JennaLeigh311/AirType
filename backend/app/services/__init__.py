"""
AirType Services Package

This package contains service modules for video processing, feature extraction,
prediction, and deduplication.
"""

from app.services.video_processor import VideoProcessor
from app.services.feature_extractor import FeatureExtractor
from app.services.predictor import Predictor
from app.services.deduplicator import Deduplicator

__all__ = ["VideoProcessor", "FeatureExtractor", "Predictor", "Deduplicator"]
