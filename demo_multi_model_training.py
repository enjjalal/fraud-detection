"""
Demonstration script for the Multi-Model Training & Optimization System.

This script provides a quick demonstration of all the key features:
- Multi-model training (XGBoost, LightGBM, CatBoost)
- Hyperparameter optimization with Optuna
- Cross-validation evaluation
- Ensemble methods
- Feature importance analysis
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from models.train_multi_models import run_training_pipeline, create_synthetic_fraud_data_splits
from models.multi_model_trainer import MultiModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_quick_training():
    """Quick demonstration with synthetic data and minimal optimization."""
    logger.info("DEMO: Quick Multi-Model Training")
    logger.info("=" * 60)
    
    # Create synthetic data for demonstration
    logger.info("Creating synthetic fraud detection data...")
    X_train, X_val, X_test, y_train, y_val, y_test = create_synthetic_fraud_data_splits()
    
    logger.info(f"Data created:")
    logger.info(f"  Training set: {X_train.shape}")
    logger.info(f"  Validation set: {X_val.shape}")
    logger.info(f"  Test set: {X_test.shape}")
    logger.info(f"  Fraud rate in training: {y_train.mean():.3%}")
    
    # Initialize trainer
    trainer = MultiModelTrainer(save_dir="models/demo")
    
    # Run pipeline with minimal optimization for demo
    logger.info("\nRunning multi-model training pipeline...")
    results = trainer.run_complete_pipeline(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        optimize_hyperparams=True,
        n_trials=5  # Minimal trials for demo
    )
    
    # Print key results
    print_demo_summary(results)
    
    return results, trainer


def print_demo_summary(results):
    """Print a concise summary of demo results."""
    print("\n" + "="*60)
    print("DEMO RESULTS SUMMARY")
    print("="*60)
    
    # Individual model performance
    if 'test_results' in results:
        print("\nIndividual Model Performance:")
        test_results = results['test_results']
        
        # Create a simple table
        print(f"{'Model':<12} {'F1-Score':<10} {'Precision':<11} {'Recall':<8} {'ROC-AUC':<8}")
        print("-" * 55)
        
        for model_name, metrics in test_results.items():
            print(f"{model_name.upper():<12} "
                  f"{metrics['f1_score']:<10.4f} "
                  f"{metrics['precision']:<11.4f} "
                  f"{metrics['recall']:<8.4f} "
                  f"{metrics['roc_auc']:<8.4f}")
    
    # Ensemble performance
    if 'ensemble_results' in results:
        print("\nEnsemble Model Performance:")
        ensemble_results = results['ensemble_results']
        
        print(f"{'Ensemble':<15} {'F1-Score':<10} {'Precision':<11} {'Recall':<8} {'ROC-AUC':<8}")
        print("-" * 60)
        
        for ensemble_name, metrics in ensemble_results.items():
            display_name = ensemble_name.replace('_', ' ').title()
            print(f"{display_name:<15} "
                  f"{metrics['f1_score']:<10.4f} "
                  f"{metrics['precision']:<11.4f} "
                  f"{metrics['recall']:<8.4f} "
                  f"{metrics['roc_auc']:<8.4f}")
    
    # Best model recommendation
    best_individual = max(results.get('test_results', {}).items(), 
                         key=lambda x: x[1]['f1_score'], default=(None, {}))
    best_ensemble = max(results.get('ensemble_results', {}).items(), 
                       key=lambda x: x[1]['f1_score'], default=(None, {}))
    
    print(f"\nBest Individual Model: {best_individual[0]} "
          f"(F1: {best_individual[1].get('f1_score', 0):.4f})")
    print(f"Best Ensemble Model: {best_ensemble[0]} "
          f"(F1: {best_ensemble[1].get('f1_score', 0):.4f})")
    
    # Feature importance top 5
    if 'feature_importance' in results and 'mean' in results['feature_importance']:
        print("\nTop 5 Most Important Features:")
        feature_importance = results['feature_importance']['mean']
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        
        for i, (feature, importance) in enumerate(sorted_features, 1):
            print(f"  {i}. {feature}: {importance:.4f}")
    
    print("\nDemo completed successfully!")
    print("="*60)


def demo_feature_importance():
    """Demonstrate feature importance analysis."""
    logger.info("🔍 DEMO: Feature Importance Analysis")
    
    # This would typically be called after training
    # For demo purposes, we'll show the structure
    print("""
    Feature Importance Analysis includes:
    
    📊 Cross-Model Comparison:
    - Extract importance from each trained model
    - Calculate mean and standard deviation across models
    - Identify most consistent important features
    
    📈 Statistical Analysis:
    - Mean importance across all models
    - Standard deviation (stability measure)
    - Coefficient of variation (relative stability)
    
    🎯 Key Insights:
    - Features with high mean importance are generally important
    - Features with low standard deviation are consistently important
    - Features with high CV may be model-specific
    """)


def demo_hyperparameter_optimization():
    """Demonstrate hyperparameter optimization process."""
    logger.info("⚙️ DEMO: Hyperparameter Optimization")
    
    print("""
    Hyperparameter Optimization with Optuna:
    
    🎯 Optimization Strategy:
    - TPE (Tree-structured Parzen Estimator) sampler
    - Median pruner for early stopping
    - Cross-validation based evaluation
    
    📊 Parameter Spaces:
    
    XGBoost (10 parameters):
    - max_depth: [3, 12]
    - learning_rate: [0.01, 0.3] (log scale)
    - n_estimators: [100, 1000]
    - subsample: [0.6, 1.0]
    - colsample_bytree: [0.6, 1.0]
    - reg_alpha, reg_lambda: [0, 10]
    - min_child_weight: [1, 10]
    - gamma: [0, 5]
    - scale_pos_weight: [1, 10]
    
    LightGBM (11 parameters):
    - num_leaves: [10, 300]
    - learning_rate: [0.01, 0.3] (log scale)
    - n_estimators: [100, 1000]
    - subsample: [0.6, 1.0]
    - colsample_bytree: [0.6, 1.0]
    - reg_alpha, reg_lambda: [0, 10]
    - min_child_samples: [5, 100]
    - max_depth: [3, 12]
    - min_split_gain: [0, 1]
    - subsample_freq: [1, 10]
    
    CatBoost (8 parameters):
    - depth: [4, 10]
    - learning_rate: [0.01, 0.3] (log scale)
    - iterations: [100, 1000]
    - l2_leaf_reg: [1, 10]
    - border_count: [32, 255]
    - bagging_temperature: [0, 1]
    - random_strength: [0, 10]
    - subsample: [0.6, 1.0]
    """)


def demo_ensemble_methods():
    """Demonstrate ensemble methods."""
    logger.info("🤝 DEMO: Ensemble Methods")
    
    print("""
    Ensemble Methods Available:
    
    🗳️ Voting Ensembles:
    - Soft Voting: Averages predicted probabilities
    - Hard Voting: Uses majority vote on predictions
    - Combines predictions from all base models
    
    🏗️ Stacking Ensemble:
    - Uses base model predictions as features
    - Trains meta-learner (Logistic Regression) on these features
    - Can capture complex interactions between base models
    
    📊 Evaluation:
    - All ensembles evaluated on same test set
    - Compared against individual model performance
    - Best ensemble automatically identified
    """)


if __name__ == "__main__":
    print("Multi-Model Training & Optimization Demo")
    print("=" * 60)
    
    try:
        # Run the main demo
        results, trainer = demo_quick_training()
        
        # Show additional demos
        print("\n" + "="*60)
        demo_hyperparameter_optimization()
        
        print("\n" + "="*60)
        demo_ensemble_methods()
        
        print("\n" + "="*60)
        demo_feature_importance()
        
        print(f"\nAll results saved to: {trainer.save_dir}")
        print("Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\nDemo failed with error: {e}")
        print("\nThis might be due to missing dependencies. Please run:")
        print("pip install -r requirements.txt")
