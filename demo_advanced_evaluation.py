"""
Demo Script for Advanced Evaluation System - Stage 4.

This script demonstrates the complete advanced evaluation pipeline including:
- ROC curves and AUC analysis
- Precision-Recall curves
- SHAP explanations for model interpretability
- Model calibration analysis
- Interactive dashboards

Usage:
    python demo_advanced_evaluation.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
warnings.filterwarnings('ignore')

# Local imports
from src.data.data_loader import FraudDataLoader
from src.data.feature_engineering import AdvancedFeatureEngineer
from src.models.multi_model_trainer import MultiModelTrainer
from src.evaluation.advanced_evaluation import AdvancedModelEvaluator, SHAPExplainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedEvaluationDemo:
    """Comprehensive demo of advanced evaluation capabilities."""
    
    def __init__(self):
        self.data_loader = FraudDataLoader()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.model_trainer = MultiModelTrainer(save_dir="models/demo_advanced")
        self.evaluator = AdvancedModelEvaluator(save_dir="evaluation_results/demo")
        self.shap_explainer = SHAPExplainer(save_dir="shap_explanations/demo")
        
        # Create directories
        Path("models/demo_advanced").mkdir(parents=True, exist_ok=True)
        Path("evaluation_results/demo").mkdir(parents=True, exist_ok=True)
        Path("shap_explanations/demo").mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.models = {}
    
    def load_and_prepare_data(self):
        """Load and prepare the fraud detection dataset."""
        logger.info("Loading and preparing fraud detection dataset...")
        
        try:
            # Load data
            train_data, test_data = self.data_loader.load_train_test_data()
            
            if train_data is None or test_data is None:
                # Generate synthetic data for demo
                logger.info("Generating synthetic fraud data for demo...")
                train_data, test_data = self._generate_synthetic_data()
            
            # Feature engineering
            logger.info("Applying advanced feature engineering...")
            train_features = self.feature_engineer.transform(train_data.drop('target', axis=1))
            test_features = self.feature_engineer.transform(test_data.drop('target', axis=1))
            
            # Prepare training and test sets
            self.X_train = train_features
            self.X_test = test_features
            self.y_train = train_data['target']
            self.y_test = test_data['target']
            self.feature_names = self.X_train.columns.tolist()
            
            logger.info(f"Data prepared successfully:")
            logger.info(f"  Training samples: {len(self.X_train):,}")
            logger.info(f"  Test samples: {len(self.X_test):,}")
            logger.info(f"  Features: {len(self.feature_names):,}")
            logger.info(f"  Fraud rate (train): {self.y_train.mean():.2%}")
            logger.info(f"  Fraud rate (test): {self.y_test.mean():.2%}")
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            # Fallback to synthetic data
            logger.info("Falling back to synthetic data generation...")
            train_data, test_data = self._generate_synthetic_data()
            self.X_train = train_data.drop('target', axis=1)
            self.X_test = test_data.drop('target', axis=1)
            self.y_train = train_data['target']
            self.y_test = test_data['target']
            self.feature_names = self.X_train.columns.tolist()
    
    def _generate_synthetic_data(self):
        """Generate synthetic fraud detection data for demo purposes."""
        np.random.seed(42)
        
        # Generate features
        n_train, n_test = 5000, 1000
        n_features = 20
        
        # Normal transactions
        normal_train = np.random.normal(0, 1, (int(n_train * 0.95), n_features))
        normal_test = np.random.normal(0, 1, (int(n_test * 0.95), n_features))
        
        # Fraudulent transactions (different distribution)
        fraud_train = np.random.normal(2, 1.5, (int(n_train * 0.05), n_features))
        fraud_test = np.random.normal(2, 1.5, (int(n_test * 0.05), n_features))
        
        # Combine data
        X_train = np.vstack([normal_train, fraud_train])
        X_test = np.vstack([normal_test, fraud_test])
        
        y_train = np.hstack([np.zeros(len(normal_train)), np.ones(len(fraud_train))])
        y_test = np.hstack([np.zeros(len(normal_test)), np.ones(len(fraud_test))])
        
        # Create feature names
        feature_names = [f'feature_{i:02d}' for i in range(n_features)]
        
        # Create DataFrames
        train_data = pd.DataFrame(X_train, columns=feature_names)
        train_data['target'] = y_train
        
        test_data = pd.DataFrame(X_test, columns=feature_names)
        test_data['target'] = y_test
        
        # Shuffle data
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        test_data = test_data.sample(frac=1).reset_index(drop=True)
        
        logger.info("Synthetic fraud detection data generated successfully")
        return train_data, test_data
    
    def train_models(self):
        """Train multiple models for comparison."""
        logger.info("Training multiple models for advanced evaluation...")
        
        # Use the comprehensive training pipeline
        results = self.model_trainer.run_comprehensive_pipeline(
            self.X_train, self.y_train,
            self.X_test, self.y_test,
            optimize_hyperparameters=True,
            n_trials=50  # Reduced for demo speed
        )
        
        # Store trained models
        self.models = self.model_trainer.base_models.copy()
        
        # Add ensemble models if available
        if hasattr(self.model_trainer, 'ensemble_trainer') and self.model_trainer.ensemble_trainer:
            ensemble_models = self.model_trainer.ensemble_trainer.ensemble_models
            self.models.update(ensemble_models)
        
        logger.info(f"Successfully trained {len(self.models)} models")
        return results
    
    def run_advanced_evaluation(self):
        """Run comprehensive advanced evaluation."""
        logger.info("Running advanced model evaluation...")
        
        # Evaluate all models
        evaluation_results = self.evaluator.evaluate_multiple_models(
            self.models, self.X_test, self.y_test
        )
        
        # Print summary results
        logger.info("\n" + "="*60)
        logger.info("ADVANCED EVALUATION RESULTS SUMMARY")
        logger.info("="*60)
        
        for model_name, results in evaluation_results.items():
            metrics = results['metrics']
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  ROC-AUC:    {metrics['roc_auc']:.4f}")
            logger.info(f"  PR-AUC:     {metrics['pr_auc']:.4f}")
            logger.info(f"  F1-Score:   {metrics['f1_score']:.4f}")
            logger.info(f"  Precision:  {metrics['precision']:.4f}")
            logger.info(f"  Recall:     {metrics['recall']:.4f}")
        
        return evaluation_results
    
    def run_shap_analysis(self):
        """Run SHAP interpretability analysis."""
        logger.info("Running SHAP interpretability analysis...")
        
        # Analyze top 2 performing models
        model_names = list(self.models.keys())[:2]
        
        for model_name in model_names:
            logger.info(f"\nAnalyzing {model_name} with SHAP...")
            
            try:
                # Create SHAP explainer
                self.shap_explainer.create_explainer(
                    self.models[model_name],
                    self.X_train.sample(min(100, len(self.X_train))),
                    model_name
                )
                
                # Calculate SHAP values for test set sample
                X_sample = self.X_test.sample(min(200, len(self.X_test)))
                shap_values = self.shap_explainer.calculate_shap_values(model_name, X_sample)
                
                # Generate SHAP plots
                logger.info(f"  Generating SHAP summary plot...")
                self.shap_explainer.plot_summary(model_name, X_sample, max_display=15)
                
                # Generate waterfall plot for first instance
                logger.info(f"  Generating SHAP waterfall plot...")
                self.shap_explainer.plot_waterfall(model_name, X_sample, instance_idx=0)
                
                # Get feature importance ranking
                importance_df = self.shap_explainer.get_feature_importance_ranking(model_name)
                logger.info(f"  Top 5 most important features:")
                for i, (_, row) in enumerate(importance_df.head(5).iterrows()):
                    feature_idx = int(row['feature'])
                    feature_name = self.feature_names[feature_idx]
                    importance = row['importance']
                    logger.info(f"    {i+1}. {feature_name}: {importance:.4f}")
                
            except Exception as e:
                logger.warning(f"SHAP analysis failed for {model_name}: {str(e)}")
                continue
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report."""
        logger.info("Generating comprehensive evaluation report...")
        
        report_path = Path("evaluation_results/demo/evaluation_report.md")
        
        with open(report_path, 'w') as f:
            f.write("# Advanced Evaluation Report - Stage 4\n\n")
            f.write(f"**Generated on:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Dataset Summary\n")
            f.write(f"- Training samples: {len(self.X_train):,}\n")
            f.write(f"- Test samples: {len(self.X_test):,}\n")
            f.write(f"- Features: {len(self.feature_names):,}\n")
            f.write(f"- Fraud rate (test): {self.y_test.mean():.2%}\n\n")
            
            f.write("## Models Evaluated\n")
            for i, model_name in enumerate(self.models.keys(), 1):
                f.write(f"{i}. {model_name}\n")
            f.write("\n")
            
            f.write("## Performance Summary\n")
            if hasattr(self, 'evaluation_results'):
                f.write("| Model | ROC-AUC | PR-AUC | F1-Score | Precision | Recall |\n")
                f.write("|-------|---------|--------|----------|-----------|--------|\n")
                
                for model_name, results in self.evaluator.evaluation_results.items():
                    metrics = results['metrics']
                    f.write(f"| {model_name} | {metrics['roc_auc']:.4f} | "
                           f"{metrics['pr_auc']:.4f} | {metrics['f1_score']:.4f} | "
                           f"{metrics['precision']:.4f} | {metrics['recall']:.4f} |\n")
            
            f.write("\n## Generated Visualizations\n")
            f.write("- ROC Curves Comparison\n")
            f.write("- Precision-Recall Curves\n")
            f.write("- Model Calibration Curves\n")
            f.write("- SHAP Summary Plots\n")
            f.write("- SHAP Waterfall Plots\n")
            f.write("- Feature Importance Analysis\n")
            
            f.write("\n## Files Generated\n")
            f.write("- `evaluation_results/demo/`: Evaluation plots and data\n")
            f.write("- `shap_explanations/demo/`: SHAP analysis plots\n")
            f.write("- `models/demo_advanced/`: Trained model files\n")
        
        logger.info(f"Evaluation report saved to: {report_path}")
    
    def run_complete_demo(self):
        """Run the complete advanced evaluation demo."""
        logger.info("Starting Advanced Evaluation Demo - Stage 4")
        logger.info("="*60)
        
        try:
            # Step 1: Load and prepare data
            self.load_and_prepare_data()
            
            # Step 2: Train models
            training_results = self.train_models()
            
            # Step 3: Run advanced evaluation
            evaluation_results = self.run_advanced_evaluation()
            
            # Step 4: Run SHAP analysis
            self.run_shap_analysis()
            
            # Step 5: Generate report
            self.generate_evaluation_report()
            
            logger.info("\n" + "="*60)
            logger.info("DEMO COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info("Generated files:")
            logger.info("  - ROC and PR curves: evaluation_results/demo/")
            logger.info("  - SHAP explanations: shap_explanations/demo/")
            logger.info("  - Evaluation report: evaluation_results/demo/evaluation_report.md")
            logger.info("  - Trained models: models/demo_advanced/")
            
            # Instructions for dashboard
            logger.info("\nTo run the interactive dashboard:")
            logger.info("  streamlit run src/dashboard/interpretability_dashboard.py")
            
        except Exception as e:
            logger.error(f"Demo failed with error: {str(e)}")
            raise


def main():
    """Main function to run the advanced evaluation demo."""
    demo = AdvancedEvaluationDemo()
    demo.run_complete_demo()


if __name__ == "__main__":
    main()
