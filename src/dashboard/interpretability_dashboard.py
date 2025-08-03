"""
Model Interpretability Dashboard for Fraud Detection System.

This module provides an interactive Streamlit dashboard for:
- Model performance comparison
- ROC and PR curve visualization
- SHAP explanations and feature importance
- Individual prediction analysis
- Model calibration analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# SHAP for interpretability
import shap
shap.initjs()

# Local imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.advanced_evaluation import AdvancedModelEvaluator, SHAPExplainer
from models.base_model import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Fraud Detection Model Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ModelDashboard:
    """Interactive dashboard for model interpretability and analysis."""
    
    def __init__(self):
        self.models = {}
        self.evaluator = None
        self.shap_explainer = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        
    def load_models_and_data(self, models_dir: str, data_path: str):
        """Load trained models and test data."""
        try:
            # Load models (this would be customized based on your model storage)
            models_path = Path(models_dir)
            if models_path.exists():
                for model_file in models_path.glob("*.pkl"):
                    model_name = model_file.stem
                    self.models[model_name] = joblib.load(model_file)
                    
            # Load test data
            if Path(data_path).exists():
                data = pd.read_csv(data_path)
                # Assuming last column is target
                self.X_test = data.iloc[:, :-1]
                self.y_test = data.iloc[:, -1]
                self.feature_names = self.X_test.columns.tolist()
                
            return True
        except Exception as e:
            st.error(f"Error loading models and data: {str(e)}")
            return False
    
    def run_dashboard(self):
        """Main dashboard interface."""
        st.title("🔍 Fraud Detection Model Interpretability Dashboard")
        st.markdown("---")
        
        # Sidebar for navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox(
            "Select Analysis Type",
            ["Model Overview", "Performance Comparison", "ROC & PR Curves", 
             "SHAP Analysis", "Individual Predictions", "Feature Analysis"]
        )
        
        # Main content based on selected page
        if page == "Model Overview":
            self.show_model_overview()
        elif page == "Performance Comparison":
            self.show_performance_comparison()
        elif page == "ROC & PR Curves":
            self.show_curves_analysis()
        elif page == "SHAP Analysis":
            self.show_shap_analysis()
        elif page == "Individual Predictions":
            self.show_individual_predictions()
        elif page == "Feature Analysis":
            self.show_feature_analysis()
    
    def show_model_overview(self):
        """Display model overview and basic statistics."""
        st.header("📊 Model Overview")
        
        if not self.models:
            st.warning("No models loaded. Please check your models directory.")
            return
            
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loaded Models")
            for i, model_name in enumerate(self.models.keys(), 1):
                st.write(f"{i}. **{model_name}**")
        
        with col2:
            st.subheader("Dataset Information")
            if self.X_test is not None:
                st.write(f"**Test samples:** {len(self.X_test):,}")
                st.write(f"**Features:** {len(self.feature_names):,}")
                st.write(f"**Fraud rate:** {self.y_test.mean():.2%}")
            else:
                st.write("No test data loaded")
        
        # Sample data preview
        if self.X_test is not None:
            st.subheader("Sample Data Preview")
            st.dataframe(self.X_test.head())
    
    def show_performance_comparison(self):
        """Display comprehensive performance comparison."""
        st.header("📈 Model Performance Comparison")
        
        if not self.models or self.X_test is None:
            st.warning("Models and test data required for performance analysis.")
            return
        
        # Initialize evaluator if not done
        if self.evaluator is None:
            self.evaluator = AdvancedModelEvaluator()
            
        # Evaluate all models
        with st.spinner("Evaluating models..."):
            results = self.evaluator.evaluate_multiple_models(
                self.models, self.X_test, self.y_test
            )
        
        # Create metrics comparison
        metrics_data = []
        for model_name, result in results.items():
            metrics = result['metrics']
            for metric_name, value in metrics.items():
                metrics_data.append({
                    'Model': model_name,
                    'Metric': metric_name.replace('_', ' ').title(),
                    'Value': value
                })
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Interactive bar chart
        fig = px.bar(
            df_metrics, 
            x='Model', 
            y='Value', 
            color='Metric',
            title='Model Performance Metrics Comparison',
            barmode='group'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics table
        st.subheader("Detailed Metrics")
        pivot_df = df_metrics.pivot(index='Model', columns='Metric', values='Value')
        st.dataframe(pivot_df.round(4))
    
    def show_curves_analysis(self):
        """Display ROC and Precision-Recall curves."""
        st.header("📊 ROC & Precision-Recall Curves")
        
        if not self.models or self.X_test is None:
            st.warning("Models and test data required for curve analysis.")
            return
        
        # Initialize evaluator if not done
        if self.evaluator is None:
            self.evaluator = AdvancedModelEvaluator()
            results = self.evaluator.evaluate_multiple_models(
                self.models, self.X_test, self.y_test
            )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("ROC Curves")
            roc_fig = self.evaluator.plot_roc_curves(save_plot=False)
            st.plotly_chart(roc_fig, use_container_width=True)
        
        with col2:
            st.subheader("Precision-Recall Curves")
            pr_fig = self.evaluator.plot_precision_recall_curves(save_plot=False)
            st.plotly_chart(pr_fig, use_container_width=True)
        
        # Calibration curves
        st.subheader("Model Calibration")
        cal_fig = self.evaluator.plot_calibration_curves(save_plot=False)
        st.plotly_chart(cal_fig, use_container_width=True)
    
    def show_shap_analysis(self):
        """Display SHAP analysis and interpretability."""
        st.header("🔍 SHAP Model Interpretability")
        
        if not self.models or self.X_test is None:
            st.warning("Models and test data required for SHAP analysis.")
            return
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model for SHAP Analysis",
            list(self.models.keys())
        )
        
        if selected_model:
            # Initialize SHAP explainer
            if self.shap_explainer is None:
                self.shap_explainer = SHAPExplainer()
            
            with st.spinner("Creating SHAP explainer..."):
                # Create explainer for selected model
                self.shap_explainer.create_explainer(
                    self.models[selected_model], 
                    self.X_test.sample(min(100, len(self.X_test))),
                    selected_model
                )
                
                # Calculate SHAP values for a sample
                sample_size = min(500, len(self.X_test))
                X_sample = self.X_test.sample(sample_size)
                shap_values = self.shap_explainer.calculate_shap_values(
                    selected_model, X_sample
                )
            
            # SHAP summary plot
            st.subheader("Feature Importance Summary")
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(shap_values, X_sample, max_display=20, show=False)
            st.pyplot(fig)
            
            # Feature importance ranking
            st.subheader("Top Features by SHAP Importance")
            importance_df = self.shap_explainer.get_feature_importance_ranking(selected_model)
            importance_df['feature_name'] = [self.feature_names[i] for i in importance_df['feature']]
            
            # Interactive bar chart for feature importance
            fig_imp = px.bar(
                importance_df.head(15),
                x='importance',
                y='feature_name',
                orientation='h',
                title='Top 15 Features by SHAP Importance'
            )
            fig_imp.update_layout(height=500)
            st.plotly_chart(fig_imp, use_container_width=True)
    
    def show_individual_predictions(self):
        """Analyze individual predictions with SHAP explanations."""
        st.header("🎯 Individual Prediction Analysis")
        
        if not self.models or self.X_test is None:
            st.warning("Models and test data required for individual analysis.")
            return
        
        # Model selection
        selected_model = st.selectbox(
            "Select Model",
            list(self.models.keys()),
            key="individual_model"
        )
        
        # Instance selection
        instance_idx = st.number_input(
            "Select Instance Index",
            min_value=0,
            max_value=len(self.X_test) - 1,
            value=0
        )
        
        if selected_model:
            model = self.models[selected_model]
            instance = self.X_test.iloc[instance_idx]
            true_label = self.y_test.iloc[instance_idx]
            
            # Get prediction
            pred_proba = model.predict_proba(instance.values.reshape(1, -1))[0]
            prediction = model.predict(instance.values.reshape(1, -1))[0]
            
            # Display prediction info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True Label", "Fraud" if true_label == 1 else "Normal")
            with col2:
                st.metric("Prediction", "Fraud" if prediction == 1 else "Normal")
            with col3:
                st.metric("Fraud Probability", f"{pred_proba[1]:.3f}")
            
            # Feature values for this instance
            st.subheader("Feature Values")
            feature_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Value': instance.values
            })
            st.dataframe(feature_df)
            
            # SHAP explanation for this instance
            if self.shap_explainer is None:
                self.shap_explainer = SHAPExplainer()
                self.shap_explainer.create_explainer(
                    model, self.X_test.sample(min(100, len(self.X_test))), selected_model
                )
            
            # Calculate SHAP values for this instance
            shap_values = self.shap_explainer.calculate_shap_values(
                selected_model, instance.values.reshape(1, -1)
            )
            
            # SHAP waterfall plot
            st.subheader("SHAP Explanation")
            fig, ax = plt.subplots(figsize=(10, 8))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.shap_explainer.explainers[selected_model].expected_value,
                    data=instance.values,
                    feature_names=self.feature_names
                ),
                show=False
            )
            st.pyplot(fig)
    
    def show_feature_analysis(self):
        """Detailed feature analysis across models."""
        st.header("🔬 Feature Analysis")
        
        if not self.models or self.X_test is None:
            st.warning("Models and test data required for feature analysis.")
            return
        
        # Feature statistics
        st.subheader("Feature Statistics")
        feature_stats = self.X_test.describe()
        st.dataframe(feature_stats)
        
        # Feature correlation heatmap
        st.subheader("Feature Correlation Matrix")
        corr_matrix = self.X_test.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
        
        # Feature distribution by class
        st.subheader("Feature Distributions by Class")
        selected_feature = st.selectbox(
            "Select Feature for Distribution Analysis",
            self.feature_names
        )
        
        if selected_feature:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot distributions for both classes
            normal_data = self.X_test[self.y_test == 0][selected_feature]
            fraud_data = self.X_test[self.y_test == 1][selected_feature]
            
            ax.hist(normal_data, alpha=0.7, label='Normal', bins=30, density=True)
            ax.hist(fraud_data, alpha=0.7, label='Fraud', bins=30, density=True)
            ax.set_xlabel(selected_feature)
            ax.set_ylabel('Density')
            ax.set_title(f'Distribution of {selected_feature} by Class')
            ax.legend()
            
            st.pyplot(fig)


def main():
    """Main function to run the dashboard."""
    dashboard = ModelDashboard()
    
    # Sidebar for data loading
    st.sidebar.header("Data Loading")
    models_dir = st.sidebar.text_input(
        "Models Directory", 
        value="models/saved"
    )
    data_path = st.sidebar.text_input(
        "Test Data Path", 
        value="data/processed/test_data.csv"
    )
    
    if st.sidebar.button("Load Models & Data"):
        success = dashboard.load_models_and_data(models_dir, data_path)
        if success:
            st.sidebar.success("Models and data loaded successfully!")
        else:
            st.sidebar.error("Failed to load models and data.")
    
    # Run the main dashboard
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
