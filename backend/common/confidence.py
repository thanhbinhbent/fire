# common/confidence.py
"""
Confidence calibration for verification results.
Implements dynamic threshold tuning based on claim characteristics.
"""

from typing import Optional
import math


class ConfidenceCalibrator:
    """
    Calibrate confidence scores dynamically based on:
    1. Claim complexity (length, entity count)
    2. Number of iterations taken
    3. Evidence quality
    """

    def __init__(self):
        # Base confidence thresholds
        self.base_threshold = 0.7  # Default confidence threshold
        self.min_threshold = 0.5
        self.max_threshold = 0.95

    def calculate_confidence(
        self,
        verdict: str,
        iterations: int,
        max_iterations: int,
        claim_length: int,
        evidence_count: int,
        evidence_quality: float = 0.5
    ) -> float:
        """
        Calculate dynamic confidence score.

        Args:
            verdict: Final verdict (SUPPORTS, REFUTES, NOT ENOUGH INFO)
            iterations: Number of iterations taken
            max_iterations: Maximum allowed iterations
            claim_length: Number of tokens in claim
            evidence_count: Number of evidence pieces found
            evidence_quality: Average quality score of evidence (0-1)

        Returns:
            float: Confidence score between 0 and 1
        """
        # Base confidence from verdict certainty
        base_scores = {
            "SUPPORTS": 0.75,
            "REFUTES": 0.75,
            "NOT ENOUGH INFO": 0.4,
            "SUPPORTED": 0.75,
            "REFUTED": 0.75,
        }
        confidence = base_scores.get(verdict.upper(), 0.5)

        # Adjust for iteration count (fewer iterations = higher confidence)
        iteration_ratio = iterations / max(max_iterations, 1)
        iteration_penalty = iteration_ratio * 0.2
        confidence -= iteration_penalty

        # Adjust for claim complexity
        # Longer claims are harder to verify → lower confidence
        complexity_factor = min(claim_length / 100, 1.0)  # Normalize to 0-1
        confidence -= complexity_factor * 0.15

        # Adjust for evidence quantity
        # More evidence = higher confidence
        evidence_bonus = min(evidence_count / 5, 1.0) * 0.2
        confidence += evidence_bonus

        # Adjust for evidence quality
        quality_bonus = evidence_quality * 0.15
        confidence += quality_bonus

        # Clamp to valid range
        return max(self.min_threshold, min(confidence, self.max_threshold))

    def is_confident(
        self,
        confidence: float,
        claim_complexity: Optional[int] = None
    ) -> bool:
        """
        Determine if confidence exceeds threshold.
        Implements the IsConfident decision node.

        Args:
            confidence: Calculated confidence score
            claim_complexity: Optional claim length for dynamic threshold

        Returns:
            bool: True if confident enough to stop, False to continue iteration
        """
        # Dynamic threshold: Longer claims need higher confidence
        threshold = self.base_threshold
        if claim_complexity:
            complexity_adjustment = min(claim_complexity / 200, 0.2)
            threshold += complexity_adjustment

        return confidence >= threshold

    def get_verdict_label(self, verdict: str, confidence: float) -> str:
        """
        Convert FIRE verdict to Vietnamese label with confidence qualifier.

        Args:
            verdict: FIRE verdict (SUPPORTS, REFUTES, NOT ENOUGH INFO)
            confidence: Confidence score

        Returns:
            str: Vietnamese verdict with confidence qualifier
        """
        # Base labels
        labels = {
            "SUPPORTS": "Đúng",
            "REFUTES": "Sai",
            "NOT ENOUGH INFO": "Chưa rõ",
            "SUPPORTED": "Đúng",
            "REFUTED": "Sai",
        }
        base_label = labels.get(verdict.upper(), "Chưa rõ")

        # Add confidence qualifier
        if confidence >= 0.85:
            qualifier = " (Rất chắc chắn)"
        elif confidence >= 0.7:
            qualifier = " (Chắc chắn)"
        elif confidence >= 0.5:
            qualifier = " (Khá chắc chắn)"
        else:
            qualifier = " (Ít chắc chắn)"

        return f"{base_label}{qualifier}"


# Global instance
calibrator = ConfidenceCalibrator()


# Testing
if __name__ == "__main__":
    print("Testing confidence calibrator...")
    
    # Test calculate_confidence
    conf = calibrator.calculate_confidence(
        verdict="SUPPORTS",
        iterations=2,
        max_iterations=5,
        claim_length=50,
        evidence_count=3,
        evidence_quality=0.8
    )
    print(f"Confidence: {conf:.3f}")
    
    # Test is_confident
    is_conf = calibrator.is_confident(conf, claim_complexity=50)
    print(f"Is confident: {is_conf}")
    
    # Test get_verdict_label
    label = calibrator.get_verdict_label("SUPPORTS", conf)
    print(f"Verdict label: {label}")
