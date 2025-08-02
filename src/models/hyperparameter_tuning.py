"""
Hyperparameter tuning using Optuna for fraud detection models.
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
import logging
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
import joblib
from pathlib import Path

from .gradient_boosting import XGBoostModel, LightGBMModel, CatBoostModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterTuner:
    """Hyperparameter tuning using Optuna."""
    
    def __init__(self, save_dir: str = "models/saved"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.study = None
        self.best_params = {}
        
    def optimize_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series,
                        n_trials: int = 100) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters."""
        
        def objective(trial):
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            }
            
            model = XGBoostModel(self.save_dir)
            model.build_model(**params)
            model.train(X_train, y_train, X_val, y_val)
            
            # Use validation F1-score as objective
            val_metrics = model.evaluate(X_val, y_val)
            return val_metrics['f1_score']
        
        logger.info("Starting XGBoost hyperparameter optimization...")
        study = optuna.create_study(direction='maximize', study_name='xgboost_tuning')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params['xgboost'] = study.best_params
        logger.info(f"Best XGBoost F1-score: {study.best_value:.4f}")
        logger.info(f"Best XGBoost params: {study.best_params}")
        
        return study.best_params
    
    def optimize_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         n_trials: int = 100) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters."""
        
        def objective(trial):
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 10)
            }
            
            model = LightGBMModel(self.save_dir)
            model.build_model(**params)
            model.train(X_train, y_train, X_val, y_val)
            
            # Use validation F1-score as objective
            val_metrics = model.evaluate(X_val, y_val)
            return val_metrics['f1_score']
        
        logger.info("Starting LightGBM hyperparameter optimization...")
        study = optuna.create_study(direction='maximize', study_name='lightgbm_tuning')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params['lightgbm'] = study.best_params
        logger.info(f"Best LightGBM F1-score: {study.best_value:.4f}")
        logger.info(f"Best LightGBM params: {study.best_params}")
        
        return study.best_params
    
    def optimize_catboost(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         n_trials: int = 100) -> Dict[str, Any]:
        """Optimize CatBoost hyperparameters."""
        
        def objective(trial):
            params = {
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'iterations': trial.suggest_int('iterations', 50, 300),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
                'random_strength': trial.suggest_float('random_strength', 0, 10)
            }
            
            model = CatBoostModel(self.save_dir)
            model.build_model(**params)
            model.train(X_train, y_train, X_val, y_val)
            
            # Use validation F1-score as objective
            val_metrics = model.evaluate(X_val, y_val)
            return val_metrics['f1_score']
        
        logger.info("Starting CatBoost hyperparameter optimization...")
        study = optuna.create_study(direction='maximize', study_name='catboost_tuning')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params['catboost'] = study.best_params
        logger.info(f"Best CatBoost F1-score: {study.best_value:.4f}")
        logger.info(f"Best CatBoost params: {study.best_params}")
        
        return study.best_params
    
    def optimize_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           n_trials: int = 50) -> Dict[str, Dict[str, Any]]:
        """Optimize hyperparameters for all models."""
        logger.info("Starting hyperparameter optimization for all models...")
        
        all_best_params = {}
        
        # Optimize each model
        try:
            all_best_params['xgboost'] = self.optimize_xgboost(
                X_train, y_train, X_val, y_val, n_trials
            )
        except Exception as e:
            logger.error(f"XGBoost optimization failed: {e}")
        
        try:
            all_best_params['lightgbm'] = self.optimize_lightgbm(
                X_train, y_train, X_val, y_val, n_trials
            )
        except Exception as e:
            logger.error(f"LightGBM optimization failed: {e}")
        
        try:
            all_best_params['catboost'] = self.optimize_catboost(
                X_train, y_train, X_val, y_val, n_trials
            )
        except Exception as e:
            logger.error(f"CatBoost optimization failed: {e}")
        
        # Save best parameters
        self._save_best_params(all_best_params)
        
        logger.info("Hyperparameter optimization completed for all models!")
        return all_best_params
    
    def train_optimized_models(self, best_params: Dict[str, Dict[str, Any]],
                              X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train models with optimized hyperparameters."""
        logger.info("Training models with optimized hyperparameters...")
        
        optimized_models = {}
        
        # Train XGBoost with best params
        if 'xgboost' in best_params:
            xgb_model = XGBoostModel(self.save_dir)
            xgb_model.build_model(**best_params['xgboost'])
            xgb_model.train(X_train, y_train, X_val, y_val)
            xgb_model.save_model("_optimized")
            optimized_models['xgboost'] = xgb_model
        
        # Train LightGBM with best params
        if 'lightgbm' in best_params:
            lgb_model = LightGBMModel(self.save_dir)
            lgb_model.build_model(**best_params['lightgbm'])
            lgb_model.train(X_train, y_train, X_val, y_val)
            lgb_model.save_model("_optimized")
            optimized_models['lightgbm'] = lgb_model
        
        # Train CatBoost with best params
        if 'catboost' in best_params:
            cb_model = CatBoostModel(self.save_dir)
            cb_model.build_model(**best_params['catboost'])
            cb_model.train(X_train, y_train, X_val, y_val)
            cb_model.save_model("_optimized")
            optimized_models['catboost'] = cb_model
        
        logger.info("All optimized models trained and saved!")
        return optimized_models
    
    def _save_best_params(self, best_params: Dict[str, Dict[str, Any]]) -> None:
        """Save best parameters to file."""
        params_path = self.save_dir / "best_hyperparameters.joblib"
        joblib.dump(best_params, params_path)
        logger.info(f"Best parameters saved to {params_path}")
    
    def load_best_params(self) -> Dict[str, Dict[str, Any]]:
        """Load best parameters from file."""
        params_path = self.save_dir / "best_hyperparameters.joblib"
        
        if params_path.exists():
            best_params = joblib.load(params_path)
            logger.info(f"Best parameters loaded from {params_path}")
            return best_params
        else:
            logger.warning("No saved hyperparameters found")
            return {}

class AutoMLPipeline:
    """Automated ML pipeline with hyperparameter tuning."""
    
    def __init__(self, save_dir: str = "models/saved"):
        self.save_dir = save_dir
        self.tuner = HyperparameterTuner(save_dir)
        self.models = {}
        self.best_model = None
        
    def run_full_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series,
                         X_test: pd.DataFrame, y_test: pd.Series,
                         n_trials: int = 50) -> Dict[str, Any]:
        """Run complete AutoML pipeline."""
        logger.info("🚀 Starting AutoML Pipeline...")
        
        # Step 1: Hyperparameter optimization
        logger.info("📊 Step 1: Hyperparameter Optimization")
        best_params = self.tuner.optimize_all_models(
            X_train, y_train, X_val, y_val, n_trials
        )
        
        # Step 2: Train optimized models
        logger.info("🏋️ Step 2: Training Optimized Models")
        self.models = self.tuner.train_optimized_models(
            best_params, X_train, y_train, X_val, y_val
        )
        
        # Step 3: Evaluate all models
        logger.info("📈 Step 3: Model Evaluation")
        results = self._evaluate_all_models(X_test, y_test)
        
        # Step 4: Select best model
        logger.info("🏆 Step 4: Best Model Selection")
        self.best_model = self._select_best_model(results)
        
        # Step 5: Create ensemble
        logger.info("🤝 Step 5: Ensemble Creation")
        ensemble_results = self._create_ensemble(X_test, y_test)
        
        pipeline_results = {
            'best_params': best_params,
            'individual_results': results,
            'best_model': self.best_model.model_name if self.best_model else None,
            'ensemble_results': ensemble_results
        }
        
        logger.info("✅ AutoML Pipeline completed successfully!")
        return pipeline_results
    
    def _evaluate_all_models(self, X_test: pd.DataFrame, 
                           y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate all trained models."""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Evaluating {name}...")
            metrics = model.evaluate(X_test, y_test)
            results[name] = metrics
            
            # Print detailed report
            model.print_evaluation_report(X_test, y_test, "Test Set")
        
        return results
    
    def _select_best_model(self, results: Dict[str, Dict[str, float]]) -> Any:
        """Select best model based on F1-score."""
        if not results:
            return None
        
        best_score = 0
        best_model_name = None
        
        for name, metrics in results.items():
            if metrics['f1_score'] > best_score:
                best_score = metrics['f1_score']
                best_model_name = name
        
        if best_model_name:
            logger.info(f"🏆 Best model: {best_model_name} (F1: {best_score:.4f})")
            return self.models[best_model_name]
        
        return None
    
    def _create_ensemble(self, X_test: pd.DataFrame, 
                        y_test: pd.Series) -> Dict[str, float]:
        """Create and evaluate ensemble model."""
        if len(self.models) < 2:
            logger.warning("Need at least 2 models for ensemble")
            return {}
        
        from .base_model import ModelEnsemble
        
        ensemble = ModelEnsemble(list(self.models.values()), ensemble_method='voting')
        ensemble_metrics = ensemble.evaluate(X_test, y_test)
        
        logger.info(f"🤝 Ensemble F1-Score: {ensemble_metrics['f1_score']:.4f}")
        
        return ensemble_metrics

def main():
    """Main function for testing hyperparameter tuning."""
    # This would typically be called from the main training script
    logger.info("Hyperparameter tuning module loaded successfully!")

if __name__ == "__main__":
    main()
