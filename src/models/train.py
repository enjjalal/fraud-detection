"""
Main training script for fraud detection models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import argparse
import json
from datetime import datetime

from data.download_data import DataDownloader
from data.preprocessing import DataPreprocessor
from models.gradient_boosting import create_all_models, compare_models
from models.hyperparameter_tuning import AutoMLPipeline
from models.base_model import ModelEnsemble

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FraudDetectionTrainer:
    """Main trainer class for fraud detection models."""
    
    def __init__(self, config: dict = None):
        self.config = config or self._get_default_config()
        self.data_downloader = DataDownloader()
        self.preprocessor = DataPreprocessor()
        self.models = {}
        self.results = {}
        
        # Create necessary directories
        Path("logs").mkdir(exist_ok=True)
        Path("models/saved").mkdir(parents=True, exist_ok=True)
        
    def _get_default_config(self) -> dict:
        """Get default training configuration."""
        return {
            'data': {
                'test_size': 0.2,
                'val_size': 0.1,
                'scaling_method': 'standard',
                'resampling_method': 'smote',
                'random_state': 42
            },
            'training': {
                'use_hyperparameter_tuning': True,
                'n_trials': 50,
                'create_ensemble': True,
                'save_models': True
            },
            'models': {
                'xgboost': True,
                'lightgbm': True,
                'catboost': True,
                'sklearn_gb': True
            }
        }
    
    def run_complete_pipeline(self) -> dict:
        """Run the complete training pipeline."""
        logger.info("🚀 Starting Complete Fraud Detection Training Pipeline")
        logger.info("="*60)
        
        try:
            # Step 1: Data Loading
            logger.info("📥 Step 1: Data Loading and Acquisition")
            df, success = self.data_downloader.load_data()
            if not success:
                raise Exception("Failed to load data")
            
            logger.info(f"✅ Data loaded successfully: {df.shape}")
            
            # Step 2: Data Preprocessing
            logger.info("🔧 Step 2: Data Preprocessing and Feature Engineering")
            data_dict = self.preprocessor.prepare_data(
                df,
                test_size=self.config['data']['test_size'],
                val_size=self.config['data']['val_size'],
                scaling_method=self.config['data']['scaling_method'],
                resampling_method=self.config['data']['resampling_method'],
                random_state=self.config['data']['random_state']
            )
            
            logger.info("✅ Data preprocessing completed")
            
            # Step 3: Model Training
            if self.config['training']['use_hyperparameter_tuning']:
                logger.info("🎯 Step 3: AutoML Pipeline with Hyperparameter Tuning")
                results = self._run_automl_pipeline(data_dict)
            else:
                logger.info("🎯 Step 3: Basic Model Training")
                results = self._run_basic_training(data_dict)
            
            # Step 4: Final Evaluation and Reporting
            logger.info("📊 Step 4: Final Evaluation and Reporting")
            final_report = self._generate_final_report(results, data_dict)
            
            logger.info("🎉 Training pipeline completed successfully!")
            return final_report
            
        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {e}")
            raise
    
    def _run_automl_pipeline(self, data_dict: dict) -> dict:
        """Run AutoML pipeline with hyperparameter tuning."""
        automl = AutoMLPipeline()
        
        results = automl.run_full_pipeline(
            X_train=data_dict['X_train'],
            y_train=data_dict['y_train'],
            X_val=data_dict['X_val'],
            y_val=data_dict['y_val'],
            X_test=data_dict['X_test'],
            y_test=data_dict['y_test'],
            n_trials=self.config['training']['n_trials']
        )
        
        self.models = automl.models
        return results
    
    def _run_basic_training(self, data_dict: dict) -> dict:
        """Run basic model training without hyperparameter tuning."""
        logger.info("Training models with default parameters...")
        
        # Create all models
        self.models = create_all_models()
        
        # Train each model
        for name, model in self.models.items():
            if self.config['models'].get(name, True):
                logger.info(f"Training {name}...")
                model.train(
                    data_dict['X_train'],
                    data_dict['y_train'],
                    data_dict['X_val'],
                    data_dict['y_val']
                )
                
                if self.config['training']['save_models']:
                    model.save_model()
        
        # Compare models
        comparison_results = compare_models(
            self.models,
            data_dict['X_test'],
            data_dict['y_test']
        )
        
        return {
            'individual_results': comparison_results.to_dict('index'),
            'best_model': comparison_results.index[0] if not comparison_results.empty else None
        }
    
    def _generate_final_report(self, results: dict, data_dict: dict) -> dict:
        """Generate final training report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'original_shape': data_dict['original_shape'],
                'final_train_shape': data_dict['final_train_shape'],
                'test_shape': data_dict['X_test'].shape,
                'feature_count': len(data_dict['feature_names'])
            },
            'config': self.config,
            'results': results
        }
        
        # Save report
        report_path = Path("models/saved/training_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"📄 Training report saved to {report_path}")
        
        # Print summary
        self._print_training_summary(report)
        
        return report
    
    def _print_training_summary(self, report: dict) -> None:
        """Print training summary."""
        print("\n" + "="*60)
        print("🎉 FRAUD DETECTION MODEL TRAINING COMPLETED")
        print("="*60)
        
        print(f"📊 Dataset Info:")
        print(f"   Original Shape: {report['data_info']['original_shape']}")
        print(f"   Training Shape: {report['data_info']['final_train_shape']}")
        print(f"   Test Shape: {report['data_info']['test_shape']}")
        print(f"   Features: {report['data_info']['feature_count']}")
        
        if 'individual_results' in report['results']:
            print(f"\n🏆 Model Performance:")
            for model, metrics in report['results']['individual_results'].items():
                print(f"   {model:12} - F1: {metrics.get('f1_score', 0):.4f}, "
                      f"AUC: {metrics.get('roc_auc', 0):.4f}")
        
        if 'best_model' in report['results']:
            print(f"\n🥇 Best Model: {report['results']['best_model']}")
        
        if 'ensemble_results' in report['results']:
            ensemble_f1 = report['results']['ensemble_results'].get('f1_score', 0)
            print(f"🤝 Ensemble F1-Score: {ensemble_f1:.4f}")
        
        print("\n✅ All models saved to 'models/saved/' directory")
        print("🚀 Ready for API deployment!")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train fraud detection models')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--quick', action='store_true', 
                       help='Quick training without hyperparameter tuning')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of hyperparameter tuning trials')
    
    args = parser.parse_args()
    
    # Load config
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Override config based on arguments
    if not config:
        trainer = FraudDetectionTrainer()
        config = trainer.config
    
    if args.quick:
        config['training']['use_hyperparameter_tuning'] = False
    
    if args.trials:
        config['training']['n_trials'] = args.trials
    
    # Initialize trainer and run pipeline
    trainer = FraudDetectionTrainer(config)
    
    try:
        results = trainer.run_complete_pipeline()
        print("\n🎉 Training completed successfully!")
        return 0
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
