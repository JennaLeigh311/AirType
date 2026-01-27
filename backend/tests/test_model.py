"""
Unit tests for AirType ML model
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock


class TestHandwritingLSTM:
    """Tests for the HandwritingLSTM model"""
    
    def test_model_creation(self):
        """Test model can be created with default parameters"""
        from app.ml.model import HandwritingLSTM
        
        model = HandwritingLSTM()
        assert model is not None
    
    def test_model_forward_pass(self):
        """Test forward pass with sample input"""
        from app.ml.model import HandwritingLSTM
        
        model = HandwritingLSTM(
            input_size=7,
            hidden_size=128,
            num_layers=2,
            num_classes=62
        )
        model.eval()
        
        # Create sample input: batch_size=2, seq_len=50, features=7
        x = torch.randn(2, 50, 7)
        
        with torch.no_grad():
            output = model(x)
        
        # Output should be (batch_size, num_classes)
        assert output.shape == (2, 62)
    
    def test_model_output_range(self):
        """Test model output is valid probability distribution"""
        from app.ml.model import HandwritingLSTM
        
        model = HandwritingLSTM()
        model.eval()
        
        x = torch.randn(1, 50, 7)
        
        with torch.no_grad():
            output = model(x)
            probs = torch.softmax(output, dim=1)
        
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
        
        # All values should be between 0 and 1
        assert (probs >= 0).all()
        assert (probs <= 1).all()
    
    def test_model_gradient_flow(self):
        """Test gradients flow through the model"""
        from app.ml.model import HandwritingLSTM
        
        model = HandwritingLSTM()
        model.train()
        
        x = torch.randn(2, 50, 7)
        target = torch.randint(0, 62, (2,))
        
        output = model(x)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_model_different_sequence_lengths(self):
        """Test model handles different sequence lengths"""
        from app.ml.model import HandwritingLSTM
        
        model = HandwritingLSTM()
        model.eval()
        
        for seq_len in [10, 50, 100, 200]:
            x = torch.randn(1, seq_len, 7)
            
            with torch.no_grad():
                output = model(x)
            
            assert output.shape == (1, 62)


class TestAttentionMechanism:
    """Tests for the attention mechanism"""
    
    def test_scaled_dot_product_attention(self):
        """Test scaled dot product attention"""
        from app.ml.model import ScaledDotProductAttention
        
        attention = ScaledDotProductAttention(d_k=64)
        
        # Create sample Q, K, V
        q = torch.randn(2, 8, 50, 64)  # batch, heads, seq_len, d_k
        k = torch.randn(2, 8, 50, 64)
        v = torch.randn(2, 8, 50, 64)
        
        output, weights = attention(q, k, v)
        
        assert output.shape == (2, 8, 50, 64)
        assert weights.shape == (2, 8, 50, 50)
        
        # Attention weights should sum to 1 along the last dimension
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 8, 50), atol=1e-5)
    
    def test_multi_head_attention(self):
        """Test multi-head attention"""
        from app.ml.model import MultiHeadAttention
        
        mha = MultiHeadAttention(d_model=256, num_heads=8)
        
        x = torch.randn(2, 50, 256)
        
        output = mha(x)
        
        assert output.shape == (2, 50, 256)


class TestFeatureExtraction:
    """Tests for feature extraction"""
    
    def test_feature_extractor_basic(self):
        """Test basic feature extraction"""
        from app.services.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        # Create sample points
        points = [
            {'x': 0.1, 'y': 0.1, 'timestamp': 1000},
            {'x': 0.2, 'y': 0.2, 'timestamp': 1050},
            {'x': 0.3, 'y': 0.3, 'timestamp': 1100},
            {'x': 0.4, 'y': 0.4, 'timestamp': 1150},
            {'x': 0.5, 'y': 0.5, 'timestamp': 1200},
        ]
        
        features = extractor.extract(points)
        
        assert features is not None
        assert len(features) > 0
    
    def test_feature_extractor_normalization(self):
        """Test that features are normalized"""
        from app.services.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        
        points = [
            {'x': 0.0, 'y': 0.0, 'timestamp': 1000},
            {'x': 0.5, 'y': 0.5, 'timestamp': 1100},
            {'x': 1.0, 'y': 1.0, 'timestamp': 1200},
        ]
        
        features = extractor.extract(points)
        
        # Normalized features should be in reasonable range
        features_array = np.array(features)
        assert np.abs(features_array).max() < 100  # Not extreme values


class TestPredictor:
    """Tests for the predictor service"""
    
    @patch('app.services.predictor.torch.load')
    def test_predictor_initialization(self, mock_load):
        """Test predictor can be initialized"""
        mock_load.return_value = {
            'model_state_dict': {},
            'config': {
                'input_size': 7,
                'hidden_size': 256,
                'num_layers': 3,
                'num_classes': 62
            }
        }
        
        from app.services.predictor import Predictor
        
        predictor = Predictor()
        assert predictor is not None
    
    def test_predictor_predict_returns_correct_format(self, mock_model):
        """Test prediction returns correct format"""
        result = mock_model.predict.return_value
        
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'alternatives' in result
        assert 'inference_time' in result
        
        assert 0 <= result['confidence'] <= 1
        assert result['inference_time'] >= 0


class TestDataset:
    """Tests for the dataset module"""
    
    def test_stroke_dataset_creation(self):
        """Test StrokeDataset can be created"""
        from app.ml.dataset import StrokeDataset
        
        # Create sample data
        samples = [
            {
                'features': np.random.randn(50, 7).tolist(),
                'label': 0
            }
            for _ in range(10)
        ]
        
        dataset = StrokeDataset(samples)
        
        assert len(dataset) == 10
    
    def test_stroke_dataset_getitem(self):
        """Test StrokeDataset __getitem__"""
        from app.ml.dataset import StrokeDataset
        
        samples = [
            {
                'features': np.random.randn(50, 7).tolist(),
                'label': i % 62
            }
            for i in range(10)
        ]
        
        dataset = StrokeDataset(samples)
        
        features, label = dataset[0]
        
        assert isinstance(features, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert features.shape == (50, 7)
    
    def test_data_augmentation(self):
        """Test data augmentation functions"""
        from app.ml.dataset import augment_stroke
        
        original = np.random.randn(50, 7)
        augmented = augment_stroke(original)
        
        # Augmented should have same shape
        assert augmented.shape == original.shape
        
        # Augmented should be different from original
        assert not np.allclose(augmented, original)
