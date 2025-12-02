"""Custom DSPy metrics for scheme extraction evaluation.

Defines metrics for tracking extraction quality, token usage,
latency, and confidence calibration.
"""

import json
import time
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

import dspy
from pydantic import ValidationError

from src.models import SchemeHeader

logger = logging.getLogger(__name__)


class SchemeExtractionMetric(dspy.Metric):
    """Composite metric for evaluating scheme extraction quality.
    
    Evaluates:
    - JSON validity (can it be parsed?)
    - Schema compliance (all required fields present?)
    - Data type correctness
    - Date format validation
    - Confidence score reasonableness
    """
    
    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Optional[Any] = None
    ) -> float:
        """Evaluate extraction quality.
        
        Args:
            example: Input example with expected output
            prediction: Model prediction
            trace: Optional execution trace
            
        Returns:
            Score between 0.0 and 1.0
        """
        try:
            # Extract prediction schemes JSON
            if hasattr(prediction, 'schemes_json'):
                schemes_json = prediction.schemes_json
            else:
                logger.warning("Prediction missing schemes_json")
                return 0.0
            
            # Score components
            scores = {}
            
            # 1. JSON validity (30% weight)
            scores['json_valid'] = self._score_json_validity(schemes_json)
            
            # 2. Schema compliance (30% weight)
            scores['schema_compliant'] = self._score_schema_compliance(schemes_json)
            
            # 3. Date format correctness (20% weight)
            scores['dates_valid'] = self._score_date_formats(schemes_json)
            
            # 4. Required fields present (20% weight)
            scores['fields_present'] = self._score_required_fields(schemes_json)
            
            # Calculate weighted average
            total_score = (
                scores['json_valid'] * 0.3 +
                scores['schema_compliant'] * 0.3 +
                scores['dates_valid'] * 0.2 +
                scores['fields_present'] * 0.2
            )
            
            logger.debug(f"Extraction metric scores: {scores}, total: {total_score:.3f}")
            return total_score
            
        except Exception as e:
            logger.error(f"Metric evaluation failed: {e}")
            return 0.0
    
    def _score_json_validity(self, schemes_json: str) -> float:
        """Check if JSON can be parsed."""
        try:
            # Clean JSON
            cleaned = schemes_json.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            data = json.loads(cleaned)
            
            # Check for schemes key
            if "schemes" in data and isinstance(data["schemes"], list):
                return 1.0
            return 0.5  # Valid JSON but wrong structure
            
        except json.JSONDecodeError:
            return 0.0
    
    def _score_schema_compliance(self, schemes_json: str) -> float:
        """Check if schemes comply with Pydantic schema."""
        try:
            cleaned = schemes_json.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            
            data = json.loads(cleaned.strip())
            
            if "schemes" not in data:
                return 0.0
            
            schemes = data["schemes"]
            if not schemes:  # Empty array is valid
                return 1.0
            
            # Try to validate each scheme
            valid_count = 0
            for scheme_data in schemes:
                try:
                    SchemeHeader(**scheme_data)
                    valid_count += 1
                except ValidationError:
                    pass
            
            return valid_count / len(schemes) if schemes else 1.0
            
        except Exception:
            return 0.0
    
    def _score_date_formats(self, schemes_json: str) -> float:
        """Check if dates are in YYYY-MM-DD format or null."""
        try:
            cleaned = schemes_json.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
            
            data = json.loads(cleaned.strip())
            schemes = data.get("schemes", [])
            
            if not schemes:
                return 1.0
            
            date_fields = [
                "duration_start_date",
                "duration_end_date",
                "starting_at",
                "ending_at",
                "price_drop_date"
            ]
            
            total_dates = 0
            valid_dates = 0
            
            for scheme in schemes:
                for field in date_fields:
                    if field in scheme and scheme[field] is not None:
                        total_dates += 1
                        date_str = scheme[field]
                        # Check YYYY-MM-DD format
                        if isinstance(date_str, str) and len(date_str) == 10:
                            try:
                                datetime.strptime(date_str, "%Y-%m-%d")
                                valid_dates += 1
                            except ValueError:
                                pass
            
            return valid_dates / total_dates if total_dates > 0 else 1.0
            
        except Exception:
            return 0.5
    
    def _score_required_fields(self, schemes_json: str) -> float:
        """Check if required fields are present."""
        try:
            data = json.loads(schemes_json.strip().strip("```").strip("json"))
            schemes = data.get("schemes", [])
            
            if not schemes:
                return 1.0
            
            required_fields = [
                "scheme_type",
                "scheme_sub_type",
                "scheme_name",
                "confidence"
            ]
            
            total_required = len(required_fields) * len(schemes)
            present_count = 0
            
            for scheme in schemes:
                for field in required_fields:
                    if field in scheme and scheme[field] is not None:
                        present_count += 1
            
            return present_count / total_required if total_required > 0 else 1.0
            
        except Exception:
            return 0.0


class TokenUsageMetric(dspy.Metric):
    """Metric for tracking token consumption."""
    
    def __init__(self, budget: Optional[int] = None):
        """Initialize token usage metric.
        
        Args:
            budget: Optional token budget threshold
        """
        self.budget = budget
        self.total_tokens = 0
        self.call_count = 0
    
    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Optional[Any] = None
    ) -> float:
        """Track token usage.
        
        Returns:
            1.0 if within budget, score based on budget utilization otherwise
        """
        # Extract token count from LM if available
        if trace and hasattr(trace, 'token_usage'):
            tokens = trace.token_usage
        else:
            # Estimate based on text length (rough approximation)
            tokens = len(str(prediction)) // 4
        
        self.total_tokens += tokens
        self.call_count += 1
        
        logger.info(f"Token usage: {tokens} (total: {self.total_tokens}, avg: {self.total_tokens / self.call_count:.1f})")
        
        if self.budget:
            utilization = tokens / self.budget
            return max(0.0, 1.0 - utilization)  # Prefer lower usage
        
        return 1.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get token usage statistics."""
        return {
            "total_tokens": self.total_tokens,
            "call_count": self.call_count,
            "avg_tokens_per_call": self.total_tokens / max(1, self.call_count),
            "budget": self.budget,
            "under_budget": self.budget is None or self.total_tokens < self.budget
        }


class LatencyMetric(dspy.Metric):
    """Metric for tracking inference latency."""
    
    def __init__(self):
        self.latencies = []
        self.start_times = {}
    
    def start(self, example_id: str):
        """Start timing for an example."""
        self.start_times[example_id] = time.time()
    
    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Optional[Any] = None
    ) -> float:
        """Record latency for this prediction.
        
        Returns:
            1.0 (latency tracking doesn't affect score)
        """
        example_id = getattr(example, 'id', str(hash(str(example))))
        
        if example_id in self.start_times:
            latency = time.time() - self.start_times[example_id]
            self.latencies.append(latency)
            logger.info(f"Inference latency: {latency:.2f}s")
            del self.start_times[example_id]
        
        return 1.0
    
    def get_stats(self) -> Dict[str, float]:
        """Get latency statistics."""
        if not self.latencies:
            return {"count": 0}
        
        return {
            "count": len(self.latencies),
            "total": sum(self.latencies),
            "mean": sum(self.latencies) / len(self.latencies),
            "min": min(self.latencies),
            "max": max(self.latencies)
        }


class ConfidenceCalibrationMetric(dspy.Metric):
    """Metric to measure if confidence scores align with actual accuracy.
    
    Tracks whether high-confidence predictions are actually more accurate
    than low-confidence ones.
    """
    
    def __init__(self):
        self.predictions = []  # List of (confidence, accuracy) tuples
    
    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Optional[Any] = None
    ) -> float:
        """Evaluate confidence calibration.
        
        If we have ground truth, compare predicted confidence to actual accuracy.
        """
        try:
            # Extract confidence from prediction
            confidence = self._extract_confidence(prediction)
            
            # If we have ground truth, calculate accuracy
            if hasattr(example, 'expected_schemes'):
                accuracy = self._calculate_accuracy(
                    prediction,
                    example.expected_schemes
                )
                self.predictions.append((confidence, accuracy))
                
                # Return how close confidence matches accuracy
                return 1.0 - abs(confidence - accuracy)
            
            # Without ground truth, just verify confidence is in valid range
            return 1.0 if 0.0 <= confidence <= 1.0 else 0.0
            
        except Exception as e:
            logger.warning(f"Confidence calibration check failed: {e}")
            return 0.5
    
    def _extract_confidence(self, prediction: dspy.Prediction) -> float:
        """Extract confidence score from prediction."""
        try:
            if hasattr(prediction, 'schemes_json'):
                data = json.loads(prediction.schemes_json.strip().strip("```").strip("json"))
                schemes = data.get("schemes", [])
                if schemes:
                    # Average confidence across all schemes
                    confidences = [s.get("confidence", 0.5) for s in schemes]
                    return sum(confidences) / len(confidences)
            return 0.5
        except Exception:
            return 0.5
    
    def _calculate_accuracy(
        self,
        prediction: dspy.Prediction,
        expected: List[Dict[str, Any]]
    ) -> float:
        """Calculate how accurate the prediction is vs expected."""
        try:
            data = json.loads(prediction.schemes_json.strip().strip("```").strip("json"))
            predicted_schemes = data.get("schemes", [])
            
            if not expected:
                return 1.0 if not predicted_schemes else 0.0
            
            # Simple accuracy: check if key fields match
            matches = 0
            for exp_scheme in expected:
                for pred_scheme in predicted_schemes:
                    if self._schemes_match(exp_scheme, pred_scheme):
                        matches += 1
                        break
            
            return matches / max(len(expected), len(predicted_schemes))
            
        except Exception:
            return 0.0
    
    def _schemes_match(self, expected: Dict, predicted: Dict) -> bool:
        """Check if two schemes match on key fields."""
        key_fields = ["scheme_type", "scheme_sub_type", "scheme_name"]
        
        for field in key_fields:
            if expected.get(field) != predicted.get(field):
                return False
        
        return True
    
    def get_calibration_stats(self) -> Dict[str, Any]:
        """Get confidence calibration statistics."""
        if not self.predictions:
            return {"count": 0}
        
        # Group by confidence bins
        bins = {
            "0.0-0.2": [],
            "0.2-0.4": [],
            "0.4-0.6": [],
            "0.6-0.8": [],
            "0.8-1.0": []
        }
        
        for conf, acc in self.predictions:
            if conf < 0.2:
                bins["0.0-0.2"].append(acc)
            elif conf < 0.4:
                bins["0.2-0.4"].append(acc)
            elif conf < 0.6:
                bins["0.4-0.6"].append(acc)
            elif conf < 0.8:
                bins["0.6-0.8"].append(acc)
            else:
                bins["0.8-1.0"].append(acc)
        
        bin_stats = {}
        for bin_name, accuracies in bins.items():
            if accuracies:
                bin_stats[bin_name] = {
                    "count": len(accuracies),
                    "avg_accuracy": sum(accuracies) / len(accuracies)
                }
        
        return {
            "total_predictions": len(self.predictions),
            "bins": bin_stats
        }


class CompositeMetric(dspy.Metric):
    """Combines multiple metrics into a single score."""
    
    def __init__(self, metrics: Dict[str, tuple]):
        """Initialize composite metric.
        
        Args:
            metrics: Dict of {name: (metric_instance, weight)}
        """
        self.metrics = metrics
    
    def __call__(
        self,
        example: dspy.Example,
        prediction: dspy.Prediction,
        trace: Optional[Any] = None
    ) -> float:
        """Evaluate all metrics and return weighted average."""
        total_score = 0.0
        total_weight = 0.0
        
        for name, (metric, weight) in self.metrics.items():
            try:
                score = metric(example, prediction, trace)
                total_score += score * weight
                total_weight += weight
                logger.debug(f"Metric '{name}': {score:.3f} (weight: {weight})")
            except Exception as e:
                logger.warning(f"Metric '{name}' failed: {e}")
        
        final_score = total_score / total_weight if total_weight > 0 else 0.0
        logger.info(f"Composite metric score: {final_score:.3f}")
        return final_score
