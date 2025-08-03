"""
Advanced Evaluation System for Fraud Detection Models.

This module provides comprehensive model evaluation capabilities including:
- ROC curves and AUC analysis
- Precision-Recall curves
- SHAP explanations for model interpretability
- Feature importance analysis
- Performance comparison dashboards
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML evaluation libraries
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, accuracy_score
)
from sklearn.calibration import calibration_curve

# SHAP for model interpretability
import shap

# Plotting libraries
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Local imports
from ..models.base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedModelEvaluator:
    """Advanced evaluation system for fraud detection models."""
    
    def __init__(self, save_dir: str = "evaluation_results"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Store evaluation results
        self.evaluation_results = {}
        self.shap_explainers = {}
        self.shap_values = {}
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def evaluate_single_model(self, model: BaseModel, X_test: pd.DataFrame, 
                             y_test: pd.Series, model_name: str = None) -> Dict[str, Any]:
        """Comprehensive evaluation of a single model."""
        if model_name is None:
            model_name = model.model_name
            
        logger.info(f"Starting advanced evaluation for {model_name}")
        
        # Get predictions and probabilities
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        if y_pred_proba.ndim > 1:
            y_pred_proba = y_pred_proba[:, 1]  # Probability of positive class
            
        # Calculate basic metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate curves data
        curves_data = self._generate_curves_data(y_test, y_pred_proba)
        
        # Store results
        self.evaluation_results[model_name] = {
            'metrics': metrics,
            'curves_data': curves_data,
            'predictions': {
                'y_pred': y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred,
                'y_pred_proba': y_pred_proba.tolist() if hasattr(y_pred_proba, 'tolist') else y_pred_proba
            }
        }
        
        logger.info(f"Evaluation completed for {model_name}")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        
        return self.evaluation_results[model_name]
    
    def evaluate_multiple_models(self, models: Dict[str, BaseModel], 
                                X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate multiple models and compare their performance."""
        logger.info(f"Evaluating {len(models)} models")
        
        for model_name, model in models.items():
            self.evaluate_single_model(model, X_test, y_test, model_name)
            
        # Generate comparison plots
        self._create_comparison_plots()
        
        return self.evaluation_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'pr_auc': average_precision_score(y_true, y_pred_proba)
        }
    
    def _generate_curves_data(self, y_true: np.ndarray, 
                             y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """Generate data for ROC and PR curves."""
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=10
        )
        
        return {
            'roc': {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist(),
                'auc': roc_auc
            },
            'pr': {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist(),
                'auc': pr_auc
            },
            'calibration': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist()
            }
        }
    
    def plot_roc_curves(self, save_plot: bool = True) -> go.Figure:
        """Plot ROC curves for all evaluated models."""
        fig = go.Figure()
        
        # Add diagonal line (random classifier)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Random Classifier'
        ))
        
        # Add ROC curves for each model
        colors = px.colors.qualitative.Set1
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            roc_data = results['curves_data']['roc']
            
            fig.add_trace(go.Scatter(
                x=roc_data['fpr'],
                y=roc_data['tpr'],
                mode='lines',
                name=f'{model_name} (AUC = {roc_data["auc"]:.3f})',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title='ROC Curves Comparison',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=800,
            height=600,
            template='plotly_white'
        )
        
        if save_plot:
            fig.write_html(self.save_dir / 'roc_curves.html')
            fig.write_image(self.save_dir / 'roc_curves.png')
            
        return fig
    
    def plot_precision_recall_curves(self, save_plot: bool = True) -> go.Figure:
        """Plot Precision-Recall curves for all evaluated models."""
        fig = go.Figure()
        
        # Add baseline (random classifier)
        baseline = np.mean([results['curves_data']['pr']['precision'][0] 
                           for results in self.evaluation_results.values()])
        fig.add_hline(y=baseline, line_dash="dash", line_color="gray", 
                     annotation_text="Random Classifier")
        
        # Add PR curves for each model
        colors = px.colors.qualitative.Set1
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            pr_data = results['curves_data']['pr']
            
            fig.add_trace(go.Scatter(
                x=pr_data['recall'],
                y=pr_data['precision'],
                mode='lines',
                name=f'{model_name} (AUC = {pr_data["auc"]:.3f})',
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title='Precision-Recall Curves Comparison',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=800,
            height=600,
            template='plotly_white'
        )
        
        if save_plot:
            fig.write_html(self.save_dir / 'pr_curves.html')
            fig.write_image(self.save_dir / 'pr_curves.png')
            
        return fig
    
    def plot_calibration_curves(self, save_plot: bool = True) -> go.Figure:
        """Plot calibration curves for all evaluated models."""
        fig = go.Figure()
        
        # Add perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Perfect Calibration'
        ))
        
        # Add calibration curves for each model
        colors = px.colors.qualitative.Set1
        for i, (model_name, results) in enumerate(self.evaluation_results.items()):
            cal_data = results['curves_data']['calibration']
            
            fig.add_trace(go.Scatter(
                x=cal_data['mean_predicted_value'],
                y=cal_data['fraction_of_positives'],
                mode='lines+markers',
                name=model_name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title='Calibration Curves Comparison',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            width=800,
            height=600,
            template='plotly_white'
        )
        
        if save_plot:
            fig.write_html(self.save_dir / 'calibration_curves.html')
            fig.write_image(self.save_dir / 'calibration_curves.png')
            
        return fig
    
    def _create_comparison_plots(self):
        """Create all comparison plots."""
        self.plot_roc_curves()
        self.plot_precision_recall_curves()
        self.plot_calibration_curves()
        self._plot_metrics_comparison()
        
    def _plot_metrics_comparison(self, save_plot: bool = True) -> go.Figure:
        """Create a comprehensive metrics comparison plot."""
        metrics_data = []
        
        for model_name, results in self.evaluation_results.items():
            metrics = results['metrics']
            for metric_name, value in metrics.items():
                metrics_data.append({
                    'Model': model_name,
                    'Metric': metric_name.replace('_', ' ').title(),
                    'Value': value
                })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        fig = px.bar(df_metrics, x='Model', y='Value', color='Metric',
                     title='Model Performance Metrics Comparison',
                     barmode='group')
        
        fig.update_layout(
            width=1000,
            height=600,
            template='plotly_white'
        )
        
        if save_plot:
            fig.write_html(self.save_dir / 'metrics_comparison.html')
            fig.write_image(self.save_dir / 'metrics_comparison.png')
            
        return fig


class SHAPExplainer:
    """SHAP-based model interpretability analysis."""
    
    def __init__(self, save_dir: str = "shap_explanations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.explainers = {}
        self.shap_values = {}
        
    def create_explainer(self, model: BaseModel, X_background: pd.DataFrame, 
                        model_name: str = None) -> Any:
        """Create SHAP explainer for a model."""
        if model_name is None:
            model_name = model.model_name
            
        logger.info(f"Creating SHAP explainer for {model_name}")
        
        # Choose appropriate explainer based on model type
        model_type = type(model.model).__name__.lower()
        
        if any(tree_type in model_type for tree_type in ['xgb', 'lgb', 'catboost', 'randomforest']):
            # Tree explainer for tree-based models
            explainer = shap.TreeExplainer(model.model)
        else:
            # Kernel explainer for other models
            explainer = shap.KernelExplainer(
                model.predict_proba, 
                X_background.sample(min(100, len(X_background)))
            )
        
        self.explainers[model_name] = explainer
        return explainer
    
    def calculate_shap_values(self, model_name: str, X_explain: pd.DataFrame) -> np.ndarray:
        """Calculate SHAP values for given data."""
        if model_name not in self.explainers:
            raise ValueError(f"No explainer found for model {model_name}")
            
        logger.info(f"Calculating SHAP values for {model_name}")
        
        explainer = self.explainers[model_name]
        shap_values = explainer.shap_values(X_explain)
        
        # Handle multi-class output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class for binary classification
            
        self.shap_values[model_name] = shap_values
        return shap_values
    
    def plot_summary(self, model_name: str, X_explain: pd.DataFrame, 
                    max_display: int = 20, save_plot: bool = True) -> None:
        """Create SHAP summary plot."""
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for model {model_name}")
            
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values[model_name], 
            X_explain,
            max_display=max_display,
            show=False
        )
        
        if save_plot:
            plt.savefig(self.save_dir / f'{model_name}_shap_summary.png', 
                       bbox_inches='tight', dpi=300)
            
        plt.show()
    
    def plot_waterfall(self, model_name: str, X_explain: pd.DataFrame, 
                      instance_idx: int = 0, save_plot: bool = True) -> None:
        """Create SHAP waterfall plot for a single instance."""
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for model {model_name}")
            
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[model_name][instance_idx],
                base_values=self.explainers[model_name].expected_value,
                data=X_explain.iloc[instance_idx].values,
                feature_names=X_explain.columns.tolist()
            ),
            show=False
        )
        
        if save_plot:
            plt.savefig(self.save_dir / f'{model_name}_waterfall_{instance_idx}.png', 
                       bbox_inches='tight', dpi=300)
            
        plt.show()
    
    def plot_force(self, model_name: str, X_explain: pd.DataFrame, 
                  instance_idx: int = 0, save_plot: bool = True) -> None:
        """Create SHAP force plot for a single instance."""
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for model {model_name}")
            
        force_plot = shap.force_plot(
            self.explainers[model_name].expected_value,
            self.shap_values[model_name][instance_idx],
            X_explain.iloc[instance_idx],
            matplotlib=True,
            show=False
        )
        
        if save_plot:
            plt.savefig(self.save_dir / f'{model_name}_force_{instance_idx}.png', 
                       bbox_inches='tight', dpi=300)
            
        return force_plot
    
    def plot_dependence(self, model_name: str, feature_name: str, 
                       X_explain: pd.DataFrame, save_plot: bool = True) -> None:
        """Create SHAP dependence plot for a feature."""
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for model {model_name}")
            
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_name,
            self.shap_values[model_name],
            X_explain,
            show=False
        )
        
        if save_plot:
            plt.savefig(self.save_dir / f'{model_name}_dependence_{feature_name}.png', 
                       bbox_inches='tight', dpi=300)
            
        plt.show()
    
    def get_feature_importance_ranking(self, model_name: str) -> pd.DataFrame:
        """Get feature importance ranking based on SHAP values."""
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for model {model_name}")
            
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values[model_name]).mean(axis=0)
        
        # Create ranking dataframe
        importance_df = pd.DataFrame({
            'feature': range(len(mean_abs_shap)),
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df


def main():
    """Example usage of the Advanced Evaluation System."""
    logger.info("Advanced Model Evaluation System initialized!")


if __name__ == "__main__":
    main()
