"""
AirType ML Package

This package contains the machine learning components for handwriting recognition.
"""

from app.ml.model import HandwritingLSTM
from app.ml.dataset import StrokeDataset

__all__ = ["HandwritingLSTM", "StrokeDataset"]
