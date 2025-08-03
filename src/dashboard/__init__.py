"""
Dashboard module for fraud detection model interpretability.

This module provides interactive dashboards for:
- Model performance visualization
- SHAP-based interpretability analysis
- Feature importance analysis
- Individual prediction explanations
"""

from .interpretability_dashboard import ModelDashboard

__all__ = ['ModelDashboard']
