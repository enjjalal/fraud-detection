"""
Training script for multi-model fraud detection system.

This script demonstrates the complete multi-model training pipeline including:
- Data loading and preprocessing
- Hyperparameter optimization with Optuna
- Cross-validation evaluation
- Ensemble training
- Feature importance analysis
- Model comparison and selection
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data.feature_engineering import AdvancedFeatureEngineer
from models.multi_model_trainer import MultiModelTrainer
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_and_prepare_data(data_path: str = "data/processed") -> tuple:
    """Load and prepare data for training."""
    logger.info("Loading and preparing data...")
    
    try:
        # Try to load processed data first
        data_dir = Path(data_path)
        
        if (data_dir / "X_train.csv").exists():
            logger.info("Loading pre-processed data...")
            X_train = pd.read_csv(data_dir / "X_train.csv")
            X_val = pd.read_csv(data_dir / "X_val.csv")
            X_test = pd.read_csv(data_dir / "X_test.csv")
            y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
            y_val = pd.read_csv(data_dir / "y_val.csv").squeeze()
            y_test = pd.read_csv(data_dir / "y_test.csv").squeeze()
            
            logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            return X_train, X_val, X_test, y_train, y_val, y_test
            
        else:
            # Load raw data and process it
            logger.info("Processing raw data...")
            
            # Try to find raw data files
            raw_data_paths = [
                "data/raw/creditcard.csv",
                "data/creditcard.csv",
                "../data/creditcard.csv"
            ]
            
            df = None
            for path in raw_data_paths:
                if Path(path).exists():
                    df = pd.read_csv(path)
                    logger.info(f"Loaded data from {path}")
                    break
                    
            if df is None:
                # Create synthetic data for demonstration
                logger.warning("No data file found. Creating synthetic data for demonstration...")
                df = create_synthetic_fraud_data()
            
            # Feature engineering
            feature_engineer = AdvancedFeatureEngineer()
            df_engineered = feature_engineer.create_advanced_features(df)
            
            # Split data
            X = df_engineered.drop('Class', axis=1)
            y = df_engineered['Class']
            
            # Train/Val/Test split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
            )
            
            # Save processed data
            data_dir.mkdir(parents=True, exist_ok=True)
            X_train.to_csv(data_dir / "X_train.csv", index=False)
            X_val.to_csv(data_dir / "X_val.csv", index=False)
            X_test.to_csv(data_dir / "X_test.csv", index=False)
            y_train.to_csv(data_dir / "y_train.csv", index=False)
            y_val.to_csv(data_dir / "y_val.csv", index=False)
            y_test.to_csv(data_dir / "y_test.csv", index=False)
            
            logger.info(f"Data shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
            return X_train, X_val, X_test, y_train, y_val, y_test
            
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.info("Creating synthetic data for demonstration...")
        return create_synthetic_fraud_data_splits()


def create_synthetic_fraud_data(n_samples: int = 10000) -> pd.DataFrame:
    """Create synthetic fraud detection data for demonstration."""
    logger.info(f"Creating synthetic fraud data with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Create features similar to credit card data
    data = {}
    
    # Time feature
    data['Time'] = np.random.randint(0, 172800, n_samples)
    
    # Amount feature (log-normal distribution)
    data['Amount'] = np.random.lognormal(mean=3, sigma=1.5, size=n_samples)
    
    # V1-V28 features (PCA components, normally distributed)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    
    # Create target variable (imbalanced)
    fraud_rate = 0.002  # 0.2% fraud rate
    n_fraud = int(n_samples * fraud_rate)
    
    y = np.zeros(n_samples)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    y[fraud_indices] = 1
    
    # Modify features for fraud cases to create patterns
    for idx in fraud_indices:
        # Fraud transactions tend to have different patterns
        data['Amount'][idx] *= np.random.uniform(0.1, 2.0)  # Different amounts
        for i in range(1, 15):  # Modify some V features
            data[f'V{i}'][idx] += np.random.normal(0, 2)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df['Class'] = y.astype(int)
    
    logger.info(f"Created synthetic data: {df.shape[0]} samples, {df['Class'].sum()} fraud cases")
    return df


def create_synthetic_fraud_data_splits() -> tuple:
    """Create synthetic data and return train/val/test splits."""
    df = create_synthetic_fraud_data(10000)
    
    # Feature engineering
    feature_engineer = AdvancedFeatureEngineer()
    df_engineered = feature_engineer.create_advanced_features(df)
    
    # Split data
    X = df_engineered.drop('Class', axis=1)
    y = df_engineered['Class']
    
    # Train/Val/Test split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_training_pipeline(optimize_hyperparams: bool = True, n_trials: int = 50):
    """Run the complete multi-model training pipeline."""
    logger.info("🚀 Starting Multi-Model Fraud Detection Training")
    logger.info("=" * 70)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    
    # Initialize trainer
    trainer = MultiModelTrainer(save_dir="models/saved")
    
    # Run complete pipeline
    results = trainer.run_complete_pipeline(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        optimize_hyperparams=optimize_hyperparams,
        n_trials=n_trials
    )
    
    # Print summary results
    print_results_summary(results)
    
    return results, trainer


def print_results_summary(results: dict):
    """Print a summary of the training results."""
    logger.info("📊 TRAINING RESULTS SUMMARY")
    logger.info("=" * 50)
    
    # Test results
    if 'test_results' in results:
        logger.info("🎯 Individual Model Performance:")
        test_results = results['test_results']
        
        for model_name, metrics in test_results.items():
            logger.info(f"  {model_name.upper()}:")
            logger.info(f"    F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall: {metrics['recall']:.4f}")
            logger.info(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info("")
    
    # Ensemble results
    if 'ensemble_results' in results:
        logger.info("🤝 Ensemble Model Performance:")
        ensemble_results = results['ensemble_results']
        
        for ensemble_name, metrics in ensemble_results.items():
            logger.info(f"  {ensemble_name.upper()}:")
            logger.info(f"    F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall: {metrics['recall']:.4f}")
            logger.info(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
            logger.info("")
    
    # Best hyperparameters
    if 'best_hyperparameters' in results:
        logger.info("⚙️ Best Hyperparameters Found:")
        best_params = results['best_hyperparameters']
        
        for model_name, params in best_params.items():
            logger.info(f"  {model_name.upper()}:")
            for param, value in params.items():
                logger.info(f"    {param}: {value}")
            logger.info("")
    
    # Feature importance top 10
    if 'feature_importance' in results:
        logger.info("📈 Top 10 Most Important Features:")
        feature_importance = results['feature_importance']
        
        if 'mean' in feature_importance:
            # Sort by mean importance
            sorted_features = sorted(
                feature_importance['mean'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            for i, (feature, importance) in enumerate(sorted_features, 1):
                logger.info(f"  {i:2d}. {feature}: {importance:.4f}")
    
    logger.info("=" * 50)


def run_quick_demo():
    """Run a quick demonstration with fewer trials."""
    logger.info("Running quick demo with reduced trials...")
    return run_training_pipeline(optimize_hyperparams=True, n_trials=10)


def run_full_optimization():
    """Run full optimization with more trials."""
    logger.info("Running full optimization...")
    return run_training_pipeline(optimize_hyperparams=True, n_trials=100)


def run_without_optimization():
    """Run training without hyperparameter optimization."""
    logger.info("Running training with default parameters...")
    return run_training_pipeline(optimize_hyperparams=False, n_trials=0)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Model Fraud Detection Training")
    parser.add_argument(
        "--mode", 
        choices=["demo", "full", "no-opt"], 
        default="demo",
        help="Training mode: demo (10 trials), full (100 trials), no-opt (no optimization)"
    )
    parser.add_argument(
        "--trials", 
        type=int, 
        default=None,
        help="Number of optimization trials (overrides mode default)"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "demo":
            n_trials = args.trials if args.trials is not None else 10
            results, trainer = run_training_pipeline(optimize_hyperparams=True, n_trials=n_trials)
        elif args.mode == "full":
            n_trials = args.trials if args.trials is not None else 100
            results, trainer = run_training_pipeline(optimize_hyperparams=True, n_trials=n_trials)
        elif args.mode == "no-opt":
            results, trainer = run_training_pipeline(optimize_hyperparams=False, n_trials=0)
        
        logger.info("✅ Training completed successfully!")
        logger.info(f"Results saved to: {trainer.save_dir}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
