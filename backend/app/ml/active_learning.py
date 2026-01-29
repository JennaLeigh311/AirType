"""
AirType Active Learning Module

Provides active learning strategies to identify and prioritize
the most valuable samples for model improvement.
"""

from typing import List, Dict, Tuple, Optional, Set
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np
import random


class ActiveLearningStrategy:
    """
    Active learning strategies for intelligent sample selection.
    
    Identifies which characters/samples are most valuable to collect
    based on model confidence, error patterns, and coverage gaps.
    """
    
    # All supported characters
    ALL_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    
    def __init__(
        self,
        min_samples_per_char: int = 20,
        target_samples_per_char: int = 100,
        confusion_weight: float = 2.0,
        low_confidence_weight: float = 1.5,
        coverage_weight: float = 1.0
    ):
        """
        Initialize active learning strategy.
        
        Args:
            min_samples_per_char: Minimum samples needed per character
            target_samples_per_char: Ideal samples per character
            confusion_weight: Weight for frequently confused characters
            low_confidence_weight: Weight for low-confidence predictions
            coverage_weight: Weight for underrepresented characters
        """
        self.min_samples_per_char = min_samples_per_char
        self.target_samples_per_char = target_samples_per_char
        self.confusion_weight = confusion_weight
        self.low_confidence_weight = low_confidence_weight
        self.coverage_weight = coverage_weight
    
    def analyze_samples(
        self,
        samples: List[Dict]
    ) -> Dict:
        """
        Analyze collected samples to identify patterns.
        
        Args:
            samples: List of training sample dictionaries with fields:
                - actual_char: Correct character
                - predicted_char: Model prediction
                - predicted_confidence: Confidence score
        
        Returns:
            Analysis dictionary with coverage, confusion, and confidence data
        """
        # Initialize tracking
        char_counts = defaultdict(int)
        char_correct = defaultdict(int)
        char_confidence_sum = defaultdict(float)
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        low_confidence_samples = []
        
        for sample in samples:
            actual = sample.get("actual_char")
            predicted = sample.get("predicted_char")
            confidence = sample.get("predicted_confidence", 0)
            
            if not actual:
                continue
            
            char_counts[actual] += 1
            char_confidence_sum[actual] += confidence
            
            if actual == predicted:
                char_correct[actual] += 1
            else:
                # Track confusion
                confusion_matrix[actual][predicted] += 1
            
            # Track low confidence
            if confidence < 0.6:
                low_confidence_samples.append({
                    "actual": actual,
                    "predicted": predicted,
                    "confidence": confidence
                })
        
        # Calculate per-character metrics
        char_metrics = {}
        for char in self.ALL_CHARS:
            count = char_counts.get(char, 0)
            correct = char_correct.get(char, 0)
            conf_sum = char_confidence_sum.get(char, 0)
            
            accuracy = (correct / count * 100) if count > 0 else 0
            avg_confidence = (conf_sum / count) if count > 0 else 0
            
            char_metrics[char] = {
                "count": count,
                "correct": correct,
                "accuracy": round(accuracy, 2),
                "avg_confidence": round(avg_confidence, 4),
                "needs_samples": count < self.min_samples_per_char,
                "coverage_ratio": min(count / self.target_samples_per_char, 1.0)
            }
        
        # Identify top confusions
        top_confusions = []
        for actual, predictions in confusion_matrix.items():
            for predicted, count in sorted(predictions.items(), key=lambda x: -x[1])[:3]:
                if count >= 2:  # At least 2 occurrences
                    top_confusions.append({
                        "actual": actual,
                        "predicted_as": predicted,
                        "count": count
                    })
        
        top_confusions.sort(key=lambda x: -x["count"])
        
        return {
            "total_samples": len(samples),
            "unique_chars_covered": len([c for c in char_counts if char_counts[c] > 0]),
            "chars_below_minimum": [c for c, m in char_metrics.items() if m["needs_samples"]],
            "char_metrics": char_metrics,
            "top_confusions": top_confusions[:20],
            "low_confidence_count": len(low_confidence_samples)
        }
    
    def suggest_characters(
        self,
        analysis: Dict,
        num_suggestions: int = 10
    ) -> List[Dict]:
        """
        Suggest which characters to practice/collect next.
        
        Args:
            analysis: Analysis dictionary from analyze_samples()
            num_suggestions: Number of characters to suggest
        
        Returns:
            List of character suggestions with priority scores
        """
        char_metrics = analysis.get("char_metrics", {})
        top_confusions = analysis.get("top_confusions", [])
        
        # Build priority scores
        priority_scores = {}
        
        for char in self.ALL_CHARS:
            metrics = char_metrics.get(char, {})
            score = 0.0
            reasons = []
            
            # Coverage score (higher for underrepresented)
            coverage = metrics.get("coverage_ratio", 0)
            if coverage < 0.2:
                coverage_score = (1 - coverage) * self.coverage_weight * 10
                score += coverage_score
                reasons.append(f"Low coverage ({metrics.get('count', 0)} samples)")
            elif coverage < 0.5:
                coverage_score = (1 - coverage) * self.coverage_weight * 5
                score += coverage_score
                reasons.append(f"Moderate coverage ({metrics.get('count', 0)} samples)")
            
            # Accuracy score (higher for low accuracy)
            count = metrics.get("count", 0)
            if count >= 5:  # Only score accuracy if we have some data
                accuracy = metrics.get("accuracy", 100)
                if accuracy < 50:
                    accuracy_score = (100 - accuracy) / 100 * 10
                    score += accuracy_score
                    reasons.append(f"Low accuracy ({accuracy}%)")
                elif accuracy < 80:
                    accuracy_score = (100 - accuracy) / 100 * 5
                    score += accuracy_score
                    reasons.append(f"Moderate accuracy ({accuracy}%)")
            
            # Confidence score
            avg_conf = metrics.get("avg_confidence", 1.0)
            if count >= 5 and avg_conf < 0.6:
                conf_score = (1 - avg_conf) * self.low_confidence_weight * 5
                score += conf_score
                reasons.append(f"Low confidence ({avg_conf:.2f})")
            
            priority_scores[char] = {
                "char": char,
                "score": round(score, 2),
                "reasons": reasons,
                "current_count": count,
                "current_accuracy": metrics.get("accuracy", 0)
            }
        
        # Add confusion bonus
        for confusion in top_confusions[:10]:
            actual = confusion["actual"]
            if actual in priority_scores:
                bonus = confusion["count"] * self.confusion_weight
                priority_scores[actual]["score"] += bonus
                priority_scores[actual]["reasons"].append(
                    f"Often confused with '{confusion['predicted_as']}' ({confusion['count']}x)"
                )
        
        # Sort by score and return top suggestions
        sorted_chars = sorted(
            priority_scores.values(),
            key=lambda x: -x["score"]
        )
        
        # Filter out chars with 0 score and take top suggestions
        suggestions = [s for s in sorted_chars if s["score"] > 0][:num_suggestions]
        
        # If not enough suggestions, add random underrepresented chars
        if len(suggestions) < num_suggestions:
            remaining = [
                s for s in sorted_chars
                if s not in suggestions and s["current_count"] < self.target_samples_per_char
            ]
            random.shuffle(remaining)
            suggestions.extend(remaining[:num_suggestions - len(suggestions)])
        
        return suggestions
    
    def get_practice_session(
        self,
        analysis: Dict,
        session_length: int = 20
    ) -> List[str]:
        """
        Generate a practice session with characters to draw.
        
        Creates a balanced sequence mixing high-priority and varied characters.
        
        Args:
            analysis: Analysis dictionary from analyze_samples()
            session_length: Number of characters in the session
        
        Returns:
            List of characters to practice
        """
        suggestions = self.suggest_characters(analysis, num_suggestions=15)
        session = []
        
        # Add high-priority characters (weighted by score)
        high_priority = [s["char"] for s in suggestions[:5] if s["score"] > 5]
        
        # Build weighted list
        weighted_chars = []
        for suggestion in suggestions:
            # Weight by score, minimum 1
            weight = max(1, int(suggestion["score"] / 2))
            weighted_chars.extend([suggestion["char"]] * weight)
        
        # Add some variety from all chars
        all_chars_list = list(self.ALL_CHARS)
        
        for _ in range(session_length):
            if weighted_chars and random.random() < 0.7:
                # 70% from weighted suggestions
                char = random.choice(weighted_chars)
            else:
                # 30% random for variety
                char = random.choice(all_chars_list)
            session.append(char)
        
        # Shuffle to avoid predictable patterns
        random.shuffle(session)
        
        return session


class IncrementalTrainingManager:
    """
    Manages incremental training to avoid retraining from scratch.
    """
    
    def __init__(
        self,
        min_new_samples: int = 50,
        max_replay_ratio: float = 0.3
    ):
        """
        Initialize incremental training manager.
        
        Args:
            min_new_samples: Minimum new samples before training
            max_replay_ratio: Maximum ratio of old samples to replay
        """
        self.min_new_samples = min_new_samples
        self.max_replay_ratio = max_replay_ratio
        self.last_training_time: Optional[datetime] = None
        self.last_sample_count: int = 0
    
    def should_train(
        self,
        current_sample_count: int,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Determine if incremental training should be triggered.
        
        Args:
            current_sample_count: Total samples in database
            force: Force training regardless of thresholds
        
        Returns:
            Tuple of (should_train, reason)
        """
        if force:
            return True, "Forced training"
        
        new_samples = current_sample_count - self.last_sample_count
        
        if new_samples < self.min_new_samples:
            return False, f"Not enough new samples ({new_samples}/{self.min_new_samples})"
        
        # Check time since last training
        if self.last_training_time:
            hours_since = (datetime.utcnow() - self.last_training_time).total_seconds() / 3600
            if hours_since < 1 and new_samples < self.min_new_samples * 2:
                return False, f"Too soon since last training ({hours_since:.1f}h)"
        
        return True, f"Ready with {new_samples} new samples"
    
    def select_training_samples(
        self,
        all_samples: List[Dict],
        new_sample_ids: Set[str]
    ) -> List[Dict]:
        """
        Select samples for incremental training.
        
        Includes all new samples plus a subset of older samples for replay.
        
        Args:
            all_samples: All available training samples
            new_sample_ids: Set of IDs for newly collected samples
        
        Returns:
            Selected samples for training
        """
        new_samples = [s for s in all_samples if s.get("id") in new_sample_ids]
        old_samples = [s for s in all_samples if s.get("id") not in new_sample_ids]
        
        # Calculate replay count
        replay_count = min(
            len(old_samples),
            int(len(new_samples) * self.max_replay_ratio)
        )
        
        # Select diverse replay samples (stratified by character)
        if replay_count > 0 and old_samples:
            # Group old samples by character
            by_char = defaultdict(list)
            for sample in old_samples:
                by_char[sample.get("actual_char", "")].append(sample)
            
            # Sample evenly from each character
            replay_samples = []
            chars = list(by_char.keys())
            per_char = max(1, replay_count // len(chars))
            
            for char in chars:
                char_samples = by_char[char]
                selected = random.sample(char_samples, min(per_char, len(char_samples)))
                replay_samples.extend(selected)
            
            # Trim to exact count
            if len(replay_samples) > replay_count:
                replay_samples = random.sample(replay_samples, replay_count)
            
            return new_samples + replay_samples
        
        return new_samples
    
    def record_training(self, sample_count: int):
        """Record that training was performed."""
        self.last_training_time = datetime.utcnow()
        self.last_sample_count = sample_count
