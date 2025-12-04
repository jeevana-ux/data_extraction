"""DSPy optimizer setup for prompt optimization.

Provides utilities for optimizing DSPy modules using few-shot examples.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import dspy
from src.llm.dspy_pipeline import SchemeExtractionCoT
from src.llm.metrics import SchemeExtractionMetric, CompositeMetric, TokenUsageMetric
from src.config import ExtractionConfig

logger = logging.getLogger(__name__)


def create_optimizer(
    optimizer_type: str = "BootstrapFewShot",
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 16
) -> dspy.teleprompt.Teleprompter:
    """Create a DSPy optimizer/teleprompter.
    
    Args:
        optimizer_type: Type of optimizer ("BootstrapFewShot", "MIPROv2", etc.)
        max_bootstrapped_demos: Maximum bootstrapped demonstrations
        max_labeled_demos: Maximum labeled demonstrations
        
    Returns:
        DSPy teleprompter instance
    """
    logger.info(f"Creating optimizer: {optimizer_type}")
    
    if optimizer_type == "BootstrapFewShot":
        return dspy.BootstrapFewShot(
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos
        )
    elif optimizer_type == "MIPROv2":
        return dspy.MIPROv2(
            num_candidates=10,
            init_temperature=1.0
        )
    else:
        logger.warning(f"Unknown optimizer type: {optimizer_type}, using BootstrapFewShot")
        return dspy.BootstrapFewShot(
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos
        )


def optimize_extraction_module(
    module: SchemeExtractionCoT,
    training_data: List[dspy.Example],
    validation_data: Optional[List[dspy.Example]] = None,
    config: Optional[ExtractionConfig] = None
) -> SchemeExtractionCoT:
    """Optimize a scheme extraction module using training data.
    
    Args:
        module: SchemeExtractionCoT module to optimize
        training_data: List of training examples
        validation_data: Optional validation examples
        config: Application configuration
        
    Returns:
        Optimized module
    """
    if not training_data:
        logger.warning("No training data provided, returning unoptimized module")
        return module
    
    logger.info(f"Optimizing module with {len(training_data)} training examples")
    
    # Create metric
    metric = SchemeExtractionMetric()
    
    # Create optimizer
    optimizer_type = config.optimizer_type if config else "BootstrapFewShot"
    max_demos = config.max_bootstrapped_demos if config else 4
    
    optimizer = create_optimizer(
        optimizer_type=optimizer_type,
        max_bootstrapped_demos=max_demos
    )
    
    try:
        # Compile/optimize the module
        logger.info("Running optimizer...")
        optimized_module = optimizer.compile(
            student=module,
            trainset=training_data,
            valset=validation_data or training_data[:len(training_data)//4],
            metric=metric
        )
        
        logger.info("Optimization complete")
        return optimized_module
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        return module


def evaluate_module(
    module: SchemeExtractionCoT,
    test_data: List[dspy.Example],
    metrics: Optional[List[dspy.Metric]] = None
) -> Dict[str, Any]:
    """Evaluate a module on test data.
    
    Args:
        module: Module to evaluate
        test_data: Test examples
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of metric results
    """
    if not test_data:
        logger.warning("No test data provided")
        return {}
    
    logger.info(f"Evaluating module on {len(test_data)} test examples")
    
    # Use default metrics if none provided
    if metrics is None:
        metrics = [SchemeExtractionMetric()]
    
    results = {}
    
    for metric in metrics:
        metric_name = metric.__class__.__name__
        logger.info(f"Computing {metric_name}...")
        
        scores = []
        for example in test_data:
            try:
                # Run module
                prediction = module(
                    mail_subject=example.mail_subject,
                    mail_body=example.mail_body
                )
                
                # Compute metric
                score = metric(example, prediction)
                scores.append(score)
                
            except Exception as e:
                logger.warning(f"Evaluation failed for example: {e}")
                continue
        
        if scores:
            avg_score = sum(scores) / len(scores)
            results[metric_name] = {
                "average": avg_score,
                "scores": scores,
                "count": len(scores)
            }
            logger.info(f"{metric_name}: {avg_score:.3f}")
    
    return results
