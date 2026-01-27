"""
AirType Model Training Script

This script handles training the LSTM handwriting recognition model
with support for data augmentation, early stopping, and checkpointing.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ml.model import HandwritingLSTM, LabelSmoothingLoss
from app.ml.dataset import (
    StrokeDataset,
    collate_fn,
    create_data_loaders,
    SyntheticStrokeGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
        
        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class Trainer:
    """
    Model trainer with support for mixed precision, gradient clipping,
    and comprehensive logging.
    """
    
    def __init__(
        self,
        model: HandwritingLSTM,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        label_smoothing: float = 0.1,
        gradient_clip: float = 1.0,
        device: str = "auto",
        checkpoint_dir: str = "./checkpoints",
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            label_smoothing: Label smoothing factor
            gradient_clip: Maximum gradient norm
            device: Device to train on ('auto', 'cuda', 'cpu', 'mps')
            checkpoint_dir: Directory for saving checkpoints
        """
        # Auto-detect device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = torch.device(device)
        logger.info(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.gradient_clip = gradient_clip
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        )
        
        # Loss function
        self.criterion = LabelSmoothingLoss(
            num_classes=model.num_classes,
            smoothing=label_smoothing,
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.training_history: List[Dict] = []
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (features, labels, lengths, mask) in enumerate(pbar):
            # Move to device
            features = features.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)
            mask = mask.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(features, lengths, mask)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip,
                )
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct / total:.4f}",
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """
        Validate the model.
        
        Returns:
            Tuple of (average loss, accuracy)
        """
        if self.val_loader is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels, lengths, mask in self.val_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)
            mask = mask.to(self.device)
            
            logits, _ = self.model(features, lengths, mask)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        num_epochs: int = 100,
        early_stopping_patience: int = 10,
        save_best: bool = True,
    ) -> Dict:
        """
        Train the model.
        
        Args:
            num_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save the best model
        
        Returns:
            Training history dictionary
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.epoch = epoch + 1
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Log metrics
            metrics = {
                "epoch": self.epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "learning_rate": current_lr,
            }
            self.training_history.append(metrics)
            
            logger.info(
                f"Epoch {self.epoch}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, "
                f"lr={current_lr:.6f}"
            )
            
            # Save best model
            if save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint("best_model.pth", is_best=True)
                logger.info(f"New best model saved (val_acc={val_acc:.4f})")
            
            # Early stopping
            if early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {self.epoch}")
                break
            
            # Save periodic checkpoint
            if self.epoch % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{self.epoch}.pth")
        
        # Save final model
        self.save_checkpoint("final_model.pth")
        
        return {
            "history": self.training_history,
            "best_val_acc": self.best_val_acc,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": self.epoch,
        }
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save a model checkpoint.
        
        Args:
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        path = self.checkpoint_dir / filename
        
        self.model.save_checkpoint(
            str(path),
            optimizer=self.optimizer,
            epoch=self.epoch,
            best_val_acc=self.best_val_acc,
            best_val_loss=self.best_val_loss,
            training_history=self.training_history,
        )
        
        logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, filename: str):
        """
        Load a model checkpoint.
        
        Args:
            filename: Checkpoint filename
        """
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        self.training_history = checkpoint.get("training_history", [])
        
        logger.info(f"Checkpoint loaded: {path}")


def generate_synthetic_data(
    num_samples: int = 10000,
    val_split: float = 0.2,
) -> Tuple[List, List, List, List]:
    """
    Generate synthetic training data.
    
    Args:
        num_samples: Total number of samples
        val_split: Validation split ratio
    
    Returns:
        Tuple of (train_strokes, train_labels, val_strokes, val_labels)
    """
    generator = SyntheticStrokeGenerator(num_points=50)
    
    # Generate data
    strokes, labels = generator.generate_batch(num_samples)
    
    # Split into train/val
    split_idx = int(num_samples * (1 - val_split))
    
    # Shuffle
    indices = np.random.permutation(num_samples)
    strokes = [strokes[i] for i in indices]
    labels = [labels[i] for i in indices]
    
    train_strokes = strokes[:split_idx]
    train_labels = labels[:split_idx]
    val_strokes = strokes[split_idx:]
    val_labels = labels[split_idx:]
    
    return train_strokes, train_labels, val_strokes, val_labels


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train AirType LSTM model")
    
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing training data")
    parser.add_argument("--synthetic-samples", type=int, default=50000,
                        help="Number of synthetic samples to generate")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0001,
                        help="Weight decay (L2 regularization)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing factor")
    parser.add_argument("--gradient-clip", type=float, default=1.0,
                        help="Gradient clipping threshold")
    parser.add_argument("--early-stopping", type=int, default=10,
                        help="Early stopping patience")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Directory for saving checkpoints")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "cpu", "mps"],
                        help="Device to train on")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of data loading workers")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    logger.info("AirType Model Training")
    logger.info(f"Arguments: {args}")
    
    # Generate or load data
    if args.data_dir is None:
        logger.info(f"Generating {args.synthetic_samples} synthetic samples...")
        train_strokes, train_labels, val_strokes, val_labels = generate_synthetic_data(
            num_samples=args.synthetic_samples,
            val_split=0.2,
        )
    else:
        # Load data from directory (implement based on your data format)
        raise NotImplementedError("Data loading from directory not implemented")
    
    logger.info(f"Training samples: {len(train_strokes)}")
    logger.info(f"Validation samples: {len(val_strokes)}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_strokes=train_strokes,
        train_labels=train_labels,
        val_strokes=val_strokes,
        val_labels=val_labels,
        batch_size=args.batch_size,
        sequence_length=50,
        num_workers=args.num_workers,
    )
    
    # Create model
    model = HandwritingLSTM(
        input_dim=7,
        embed_dim=64,
        hidden_dim_1=64,
        hidden_dim_2=128,
        num_attention_heads=4,
        fc_dim=256,
        num_classes=62,
        dropout=0.3,
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        gradient_clip=args.gradient_clip,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    results = trainer.train(
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping,
        save_best=True,
    )
    
    # Save results
    results_path = Path(args.checkpoint_dir) / "training_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Training complete!")
    logger.info(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
