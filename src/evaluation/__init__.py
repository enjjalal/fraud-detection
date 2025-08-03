"""
Evaluation module for fraud detection models.

This module provides comprehensive evaluation capabilities including:
- Advanced model evaluation with ROC/PR curves
- SHAP-based model interpretability
- Performance comparison tools
- Interactive dashboards
"""

from .advanced_evaluation import AdvancedModelEvaluator, SHAPExplainer

__all__ = ['AdvancedModelEvaluator', 'SHAPExplainer']
