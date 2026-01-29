"""
AirType Fast Training Module

Provides optimized training methods including:
- GPU acceleration detection and usage
- Incremental/fine-tuning from existing model
- Batch training with augmentation
- Quick validation feedback
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ml.model import HandwritingLSTM
from app.ml.augmentation import BatchAugmentor, StrokeAugmentor
from app.ml.active_learning import IncrementalTrainingManager

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Get the best available device for training.
    
    Priority: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device for training
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        return device
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon MPS acceleration")
        return device
    else:
        logger.info("Using CPU (no GPU acceleration available)")
        return torch.device("cpu")


class FastTrainer:
    """
    Optimized trainer for quick model updates.
    
    Supports:
    - GPU acceleration (CUDA/MPS)
    - Incremental training
    - Data augmentation
    - Quick validation
    """
    
    # Character mappings
    CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
    IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        checkpoint_dir: str = "./models"
    ):
        """
        Initialize fast trainer.
        
        Args:
            model_path: Path to existing model weights (for fine-tuning)
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto')
            checkpoint_dir: Directory to save model checkpoints
        """
        # Setup device
        if device and device != "auto":
            self.device = torch.device(device)
        else:
            self.device = get_device()
        
        # Initialize model
        self.model = HandwritingLSTM(num_classes=62).to(self.device)
        
        # Load existing weights if provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        
        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.training_history: List[Dict] = []
        self.best_accuracy = 0.0
        
        # Augmentor
        self.augmentor = BatchAugmentor()
        
        # Incremental training manager
        self.incremental_manager = IncrementalTrainingManager()
    
    def _load_model(self, path: str):
        """Load model weights from file."""
        try:
            checkpoint = torch.load(path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded model from {path}")
                if "accuracy" in checkpoint:
                    self.best_accuracy = checkpoint["accuracy"]
            else:
                self.model.load_state_dict(checkpoint)
                logger.info(f"Loaded model weights from {path}")
        except Exception as e:
            logger.warning(f"Failed to load model from {path}: {e}")
    
    def prepare_data(
        self,
        features: List[List[List[float]]],
        labels: List[str],
        augment: bool = True,
        augmentation_factor: int = 5,
        sequence_length: int = 50,
        val_split: float = 0.2
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data loaders for training.
        
        Args:
            features: List of stroke feature arrays
            labels: List of character labels
            augment: Whether to apply data augmentation
            augmentation_factor: Number of augmented versions per sample
            sequence_length: Fixed sequence length for padding/truncation
            val_split: Fraction of data for validation
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Apply augmentation
        if augment:
            logger.info(f"Augmenting {len(features)} samples with factor {augmentation_factor}")
            features, labels = self.augmentor.augment_batch(
                features, labels, augmentation_factor
            )
            logger.info(f"Augmented to {len(features)} samples")
        
        # Convert features to tensors
        processed_features = []
        processed_labels = []
        
        # Model expects 7 features: x, y, vx, vy, acc, curv, pen
        expected_features = 7
        
        for feat, label in zip(features, labels):
            if label not in self.CHAR_TO_IDX:
                continue
            
            # Convert to numpy
            feat_arr = np.array(feat)
            
            # Handle 1D array (single feature per point)
            if feat_arr.ndim == 1:
                feat_arr = feat_arr.reshape(-1, 1)
            
            # Ensure we have the right number of features (pad with zeros if needed)
            current_features = feat_arr.shape[1] if feat_arr.ndim > 1 else 1
            if current_features < expected_features:
                # Pad with zeros to get 7 features
                padded_feat = np.zeros((len(feat_arr), expected_features))
                padded_feat[:, :current_features] = feat_arr
                feat_arr = padded_feat
            elif current_features > expected_features:
                # Take only the first 7 features
                feat_arr = feat_arr[:, :expected_features]
            
            # Pad or truncate to sequence_length
            if len(feat_arr) < sequence_length:
                padded = np.zeros((sequence_length, expected_features))
                padded[:len(feat_arr)] = feat_arr
                feat_arr = padded
            elif len(feat_arr) > sequence_length:
                # Resample to sequence_length
                indices = np.linspace(0, len(feat_arr) - 1, sequence_length, dtype=int)
                feat_arr = feat_arr[indices]
            
            processed_features.append(feat_arr)
            processed_labels.append(self.CHAR_TO_IDX[label])
        
        if not processed_features:
            raise ValueError("No valid samples after processing")
        
        # Convert to tensors
        X = torch.FloatTensor(np.array(processed_features)).to(self.device)
        y = torch.LongTensor(processed_labels).to(self.device)
        
        # Split into train/val
        n_samples = len(X)
        n_val = int(n_samples * val_split)
        
        # Shuffle indices
        indices = torch.randperm(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]
        
        # Create datasets
        train_dataset = TensorDataset(X[train_indices], y[train_indices])
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(32, len(train_indices)),
            shuffle=True,
            drop_last=False
        )
        
        val_loader = None
        if n_val > 0:
            val_dataset = TensorDataset(X[val_indices], y[val_indices])
            val_loader = DataLoader(
                val_dataset,
                batch_size=min(64, n_val),
                shuffle=False
            )
        
        logger.info(f"Prepared {len(train_indices)} train, {n_val} val samples")
        return train_loader, val_loader
    
    def train(
        self,
        features: List[List[List[float]]],
        labels: List[str],
        epochs: int = 10,
        learning_rate: float = 0.001,
        augment: bool = True,
        augmentation_factor: int = 5,
        early_stop_patience: int = 3
    ) -> Dict:
        """
        Train model on provided data.
        
        Args:
            features: List of stroke feature arrays
            labels: List of character labels
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            augment: Whether to apply data augmentation
            augmentation_factor: Augmentation multiplier
            early_stop_patience: Epochs without improvement before stopping
        
        Returns:
            Training results dictionary
        """
        start_time = datetime.now()
        
        # Prepare data
        train_loader, val_loader = self.prepare_data(
            features, labels,
            augment=augment,
            augmentation_factor=augmentation_factor
        )
        
        # Setup optimizer with OneCycleLR for fast convergence
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.0001
        )
        
        total_steps = epochs * len(train_loader)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=learning_rate * 10,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        history = []
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                # Ensure batch_x has correct shape (batch, seq_len, features)
                if batch_x.dim() == 2:
                    # If 2D, add feature dimension
                    batch_x = batch_x.unsqueeze(-1)
                
                batch_size = batch_x.size(0)
                seq_len = batch_x.size(1)
                
                # Forward pass (create proper lengths/masks for model)
                lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)
                mask = torch.ones(batch_size, seq_len, device=self.device)
                
                logits, _ = self.model(batch_x, lengths, mask)
                loss = criterion(logits, batch_y)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                train_correct += (predicted == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            train_acc = train_correct / train_total * 100
            avg_train_loss = train_loss / len(train_loader)
            
            # Validate
            val_acc = 0.0
            val_loss = 0.0
            if val_loader:
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        # Ensure batch_x has correct shape
                        if batch_x.dim() == 2:
                            batch_x = batch_x.unsqueeze(-1)
                        
                        batch_size = batch_x.size(0)
                        seq_len = batch_x.size(1)
                        
                        lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=self.device)
                        mask = torch.ones(batch_size, seq_len, device=self.device)
                        
                        logits, _ = self.model(batch_x, lengths, mask)
                        loss = criterion(logits, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(logits, 1)
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)
                
                val_acc = val_correct / val_total * 100
                avg_val_loss = val_loss / len(val_loader)
            
            # Log progress
            epoch_stats = {
                "epoch": epoch + 1,
                "train_loss": round(avg_train_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_loss": round(avg_val_loss, 4) if val_loader else None,
                "val_acc": round(val_acc, 2) if val_loader else None,
                "lr": scheduler.get_last_lr()[0]
            }
            history.append(epoch_stats)
            logger.info(
                f"Epoch {epoch+1}/{epochs}: "
                f"Train Loss={avg_train_loss:.4f}, Acc={train_acc:.1f}% | "
                f"Val Acc={val_acc:.1f}%"
            )
            
            # Early stopping check
            if val_loader:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    self._save_checkpoint(val_acc, "best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= early_stop_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
        
        # Save final model
        final_acc = val_acc if val_loader else train_acc
        self._save_checkpoint(final_acc, "latest_model.pt")
        
        # Record training
        self.incremental_manager.record_training(len(features))
        
        # Calculate training time
        duration = (datetime.now() - start_time).total_seconds()
        
        result = {
            "status": "completed",
            "epochs_run": len(history),
            "final_train_accuracy": history[-1]["train_acc"],
            "final_val_accuracy": history[-1]["val_acc"] if val_loader else None,
            "best_val_accuracy": best_val_acc if val_loader else None,
            "training_time_seconds": round(duration, 2),
            "samples_used": len(features),
            "device": str(self.device),
            "history": history
        }
        
        self.training_history.append(result)
        return result
    
    def fine_tune(
        self,
        features: List[List[List[float]]],
        labels: List[str],
        epochs: int = 5,
        learning_rate: float = 0.0001
    ) -> Dict:
        """
        Fine-tune model on new data with lower learning rate.
        
        Uses smaller learning rate and less augmentation for 
        incremental updates without catastrophic forgetting.
        
        Args:
            features: New training samples
            labels: Labels for new samples
            epochs: Number of fine-tuning epochs
            learning_rate: Learning rate (should be lower than initial training)
        
        Returns:
            Training results dictionary
        """
        return self.train(
            features, labels,
            epochs=epochs,
            learning_rate=learning_rate,
            augment=True,
            augmentation_factor=3,  # Less augmentation for fine-tuning
            early_stop_patience=2
        )
    
    def _save_checkpoint(self, accuracy: float, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "accuracy": accuracy,
            "timestamp": datetime.now().isoformat(),
            "device": str(self.device)
        }
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path} (acc={accuracy:.2f}%)")
    
    def evaluate(
        self,
        features: List[List[List[float]]],
        labels: List[str]
    ) -> Dict:
        """
        Evaluate model on test data.
        
        Args:
            features: Test feature arrays
            labels: Test labels
        
        Returns:
            Evaluation metrics dictionary
        """
        # Prepare data without augmentation
        test_loader, _ = self.prepare_data(
            features, labels,
            augment=False,
            val_split=0.0  # Use all data for testing
        )
        
        self.model.eval()
        correct = 0
        total = 0
        per_class_correct = {}
        per_class_total = {}
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                lengths = torch.full((batch_x.size(0),), batch_x.size(1), device=self.device)
                mask = torch.ones(batch_x.size(0), batch_x.size(1), device=self.device)
                
                logits, _ = self.model(batch_x, lengths, mask)
                _, predicted = torch.max(logits, 1)
                
                for pred, actual in zip(predicted.cpu().numpy(), batch_y.cpu().numpy()):
                    char = self.IDX_TO_CHAR[actual]
                    per_class_total[char] = per_class_total.get(char, 0) + 1
                    if pred == actual:
                        correct += 1
                        per_class_correct[char] = per_class_correct.get(char, 0) + 1
                    total += 1
        
        overall_acc = correct / total * 100 if total > 0 else 0
        
        per_class_acc = {
            char: round(per_class_correct.get(char, 0) / per_class_total[char] * 100, 2)
            for char in per_class_total
        }
        
        return {
            "overall_accuracy": round(overall_acc, 2),
            "total_samples": total,
            "correct": correct,
            "per_class_accuracy": per_class_acc
        }
    
    def get_training_info(self) -> Dict:
        """Get information about training capabilities."""
        return {
            "device": str(self.device),
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "best_accuracy": self.best_accuracy,
            "training_runs": len(self.training_history),
            "model_path": str(self.checkpoint_dir / "latest_model.pt")
        }


# Singleton instance for API use
_trainer_instance: Optional[FastTrainer] = None


def get_trainer() -> FastTrainer:
    """Get or create singleton trainer instance."""
    global _trainer_instance
    if _trainer_instance is None:
        model_dir = Path(__file__).parent.parent.parent / "models"
        model_path = model_dir / "latest_model.pt"
        _trainer_instance = FastTrainer(
            model_path=str(model_path) if model_path.exists() else None,
            checkpoint_dir=str(model_dir)
        )
    return _trainer_instance
