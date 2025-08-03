"""
Multi-Model Training & Optimization System for Fraud Detection.

This module implements comprehensive training pipelines with:
- Ensemble methods (voting, stacking, blending)
- Automated hyperparameter tuning with Optuna
- Cross-validation pipelines
- Feature importance analysis across models
- Model comparison and selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, cross_validate
)
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    classification_report, confusion_matrix
)
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

# Local imports
from .gradient_boosting import XGBoostModel, LightGBMModel, CatBoostModel
from .base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossValidationPipeline:
    """Cross-validation pipeline for model evaluation."""
    
    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=random_state
        )
        
    def evaluate_model(self, model: BaseModel, X: pd.DataFrame, y: pd.Series,
                      scoring: List[str] = None) -> Dict[str, Any]:
        """Evaluate model using cross-validation."""
        if scoring is None:
            scoring = ['f1', 'precision', 'recall', 'roc_auc']
            
        logger.info(f"Running {self.n_splits}-fold CV for {model.model_name}")
        
        # Build model if not already built
        if model.model is None:
            model.build_model()
            
        # Perform cross-validation
        cv_results = cross_validate(
            model.model, X, y, 
            cv=self.cv, 
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )
        
        # Calculate statistics
        results = {}
        for metric in scoring:
            test_scores = cv_results[f'test_{metric}']
            train_scores = cv_results[f'train_{metric}']
            
            results[f'{metric}_test_mean'] = np.mean(test_scores)
            results[f'{metric}_test_std'] = np.std(test_scores)
            results[f'{metric}_train_mean'] = np.mean(train_scores)
            results[f'{metric}_train_std'] = np.std(train_scores)
            results[f'{metric}_test_scores'] = test_scores.tolist()
            
        results['fit_time_mean'] = np.mean(cv_results['fit_time'])
        results['score_time_mean'] = np.mean(cv_results['score_time'])
        
        logger.info(f"CV Results for {model.model_name}:")
        logger.info(f"  F1: {results['f1_test_mean']:.4f} ± {results['f1_test_std']:.4f}")
        logger.info(f"  Precision: {results['precision_test_mean']:.4f} ± {results['precision_test_std']:.4f}")
        logger.info(f"  Recall: {results['recall_test_mean']:.4f} ± {results['recall_test_std']:.4f}")
        logger.info(f"  ROC-AUC: {results['roc_auc_test_mean']:.4f} ± {results['roc_auc_test_std']:.4f}")
        
        return results


class FeatureImportanceAnalyzer:
    """Analyze and compare feature importance across models."""
    
    def __init__(self):
        self.importance_data = {}
        
    def extract_importance(self, model: BaseModel, feature_names: List[str]) -> Dict[str, float]:
        """Extract feature importance from a trained model."""
        if not model.is_trained:
            logger.warning(f"Model {model.model_name} is not trained")
            return {}
            
        importance_dict = {}
        
        try:
            if hasattr(model.model, 'feature_importances_'):
                # For XGBoost, LightGBM, CatBoost
                importances = model.model.feature_importances_
                importance_dict = dict(zip(feature_names, importances))
                
            elif hasattr(model.model, 'get_feature_importance'):
                # CatBoost specific method
                importances = model.model.get_feature_importance()
                importance_dict = dict(zip(feature_names, importances))
                
            else:
                logger.warning(f"Cannot extract feature importance from {model.model_name}")
                
        except Exception as e:
            logger.error(f"Error extracting importance from {model.model_name}: {e}")
            
        return importance_dict
    
    def analyze_all_models(self, models: Dict[str, BaseModel], 
                          feature_names: List[str]) -> pd.DataFrame:
        """Analyze feature importance across all models."""
        importance_data = {}
        
        for model_name, model in models.items():
            importance_dict = self.extract_importance(model, feature_names)
            if importance_dict:
                importance_data[model_name] = importance_dict
                
        if not importance_data:
            logger.warning("No feature importance data extracted")
            return pd.DataFrame()
            
        # Create DataFrame
        df_importance = pd.DataFrame(importance_data).fillna(0)
        
        # Add statistics
        df_importance['mean'] = df_importance.mean(axis=1)
        df_importance['std'] = df_importance.std(axis=1)
        df_importance['cv'] = df_importance['std'] / df_importance['mean']
        
        # Sort by mean importance
        df_importance = df_importance.sort_values('mean', ascending=False)
        
        self.importance_data = df_importance
        
        logger.info("Feature importance analysis completed")
        logger.info(f"Top 10 most important features:")
        for i, (feature, row) in enumerate(df_importance.head(10).iterrows()):
            logger.info(f"  {i+1}. {feature}: {row['mean']:.4f} ± {row['std']:.4f}")
            
        return df_importance


class EnsembleTrainer:
    """Train and evaluate ensemble models."""
    
    def __init__(self, base_models: Dict[str, BaseModel]):
        self.base_models = base_models
        self.ensemble_models = {}
        
    def create_voting_ensemble(self, voting: str = 'soft') -> VotingClassifier:
        """Create voting ensemble from base models."""
        estimators = []
        
        for name, model in self.base_models.items():
            if model.is_trained and model.model is not None:
                estimators.append((name, model.model))
                
        if len(estimators) < 2:
            raise ValueError("Need at least 2 trained models for ensemble")
            
        voting_ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
            n_jobs=-1
        )
        
        logger.info(f"Created voting ensemble with {len(estimators)} models")
        return voting_ensemble
    
    def create_stacking_ensemble(self, meta_learner=None) -> StackingClassifier:
        """Create stacking ensemble with meta-learner."""
        if meta_learner is None:
            meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            
        estimators = []
        for name, model in self.base_models.items():
            if model.is_trained and model.model is not None:
                estimators.append((name, model.model))
                
        if len(estimators) < 2:
            raise ValueError("Need at least 2 trained models for ensemble")
            
        stacking_ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=-1
        )
        
        logger.info(f"Created stacking ensemble with {len(estimators)} models")
        return stacking_ensemble
    
    def train_all_ensembles(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Train all ensemble methods."""
        ensembles = {}
        
        try:
            # Voting ensemble (soft)
            voting_soft = self.create_voting_ensemble(voting='soft')
            voting_soft.fit(X_train, y_train)
            ensembles['voting_soft'] = voting_soft
            logger.info("Soft voting ensemble trained")
            
            # Voting ensemble (hard)
            voting_hard = self.create_voting_ensemble(voting='hard')
            voting_hard.fit(X_train, y_train)
            ensembles['voting_hard'] = voting_hard
            logger.info("Hard voting ensemble trained")
            
            # Stacking ensemble
            stacking = self.create_stacking_ensemble()
            stacking.fit(X_train, y_train)
            ensembles['stacking'] = stacking
            logger.info("Stacking ensemble trained")
            
        except Exception as e:
            logger.error(f"Error training ensembles: {e}")
            
        self.ensemble_models = ensembles
        return ensembles
    
    def evaluate_ensembles(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate all ensemble models."""
        results = {}
        
        for name, ensemble in self.ensemble_models.items():
            try:
                y_pred = ensemble.predict(X_test)
                y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
                
                metrics = {
                    'f1_score': f1_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                results[name] = metrics
                
                logger.info(f"Ensemble {name} results:")
                logger.info(f"  F1: {metrics['f1_score']:.4f}")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall: {metrics['recall']:.4f}")
                logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating ensemble {name}: {e}")
                
        return results


class AdvancedHyperparameterTuner:
    """Advanced hyperparameter tuning with Optuna."""
    
    def __init__(self, save_dir: str = "models/saved", n_jobs: int = -1):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.n_jobs = n_jobs
        self.studies = {}
        self.best_params = {}
        
    def optimize_with_cv(self, model_type: str, X: pd.DataFrame, y: pd.Series,
                        n_trials: int = 100, cv_folds: int = 5) -> Dict[str, Any]:
        """Optimize hyperparameters using cross-validation."""
        
        def objective(trial):
            # Get model-specific parameter suggestions
            if model_type == 'xgboost':
                params = self._suggest_xgboost_params(trial)
                model = XGBoostModel(self.save_dir)
            elif model_type == 'lightgbm':
                params = self._suggest_lightgbm_params(trial)
                model = LightGBMModel(self.save_dir)
            elif model_type == 'catboost':
                params = self._suggest_catboost_params(trial)
                model = CatBoostModel(self.save_dir)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
                
            # Build model with suggested parameters
            model.build_model(**params)
            
            # Cross-validation evaluation
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            scores = cross_val_score(
                model.model, X, y, 
                cv=cv, 
                scoring='f1',
                n_jobs=self.n_jobs
            )
            
            return np.mean(scores)
        
        # Create study
        study_name = f"{model_type}_cv_optimization"
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            sampler=sampler,
            pruner=pruner
        )
        
        logger.info(f"Starting {model_type} optimization with {n_trials} trials...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.studies[model_type] = study
        self.best_params[model_type] = study.best_params
        
        logger.info(f"Best {model_type} CV F1-score: {study.best_value:.4f}")
        logger.info(f"Best {model_type} params: {study.best_params}")
        
        return study.best_params
    
    def _suggest_xgboost_params(self, trial) -> Dict[str, Any]:
        """Suggest XGBoost hyperparameters."""
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10)
        }
    
    def _suggest_lightgbm_params(self, trial) -> Dict[str, Any]:
        """Suggest LightGBM hyperparameters."""
        return {
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_split_gain': trial.suggest_float('min_split_gain', 0, 1),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 10)
        }
    
    def _suggest_catboost_params(self, trial) -> Dict[str, Any]:
        """Suggest CatBoost hyperparameters."""
        return {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_float('random_strength', 0, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }


class MultiModelTrainer:
    """Main class for multi-model training and optimization."""
    
    def __init__(self, save_dir: str = "models/saved"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.cv_pipeline = CrossValidationPipeline()
        self.feature_analyzer = FeatureImportanceAnalyzer()
        self.hyperparameter_tuner = AdvancedHyperparameterTuner(save_dir)
        
        # Model storage
        self.base_models = {}
        self.ensemble_trainer = None
        self.results = {}
        
    def run_complete_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series,
                            X_test: pd.DataFrame, y_test: pd.Series,
                            optimize_hyperparams: bool = True,
                            n_trials: int = 100) -> Dict[str, Any]:
        """Run the complete multi-model training pipeline."""
        
        logger.info("Starting Multi-Model Training Pipeline")
        logger.info("=" * 60)
        
        pipeline_results = {
            'timestamp': datetime.now().isoformat(),
            'data_shapes': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            }
        }
        
        # Step 1: Hyperparameter Optimization (if enabled)
        if optimize_hyperparams:
            logger.info("Step 1: Hyperparameter Optimization")
            best_params = self._optimize_all_models(X_train, y_train, n_trials)
            pipeline_results['best_hyperparameters'] = best_params
        else:
            logger.info("Step 1: Using default hyperparameters")
            best_params = {}
            
        # Step 2: Train Base Models
        logger.info("Step 2: Training Base Models")
        self._train_base_models(X_train, y_train, X_val, y_val, best_params)
        
        # Step 3: Cross-Validation Evaluation
        logger.info("Step 3: Cross-Validation Evaluation")
        cv_results = self._evaluate_models_with_cv(X_train, y_train)
        pipeline_results['cv_results'] = cv_results
        
        # Step 4: Feature Importance Analysis
        logger.info("Step 4: Feature Importance Analysis")
        feature_importance = self._analyze_feature_importance(X_train.columns.tolist())
        pipeline_results['feature_importance'] = feature_importance.to_dict()
        
        # Step 5: Ensemble Training
        logger.info("Step 5: Ensemble Training")
        ensemble_results = self._train_and_evaluate_ensembles(X_train, y_train, X_test, y_test)
        pipeline_results['ensemble_results'] = ensemble_results
        
        # Step 6: Final Model Evaluation
        logger.info("Step 6: Final Model Evaluation")
        test_results = self._evaluate_all_models_on_test(X_test, y_test)
        pipeline_results['test_results'] = test_results
        
        # Save results
        self._save_pipeline_results(pipeline_results)
        
        logger.info("Multi-Model Training Pipeline Completed!")
        return pipeline_results
    
    def _optimize_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                           n_trials: int) -> Dict[str, Dict[str, Any]]:
        """Optimize hyperparameters for all models."""
        model_types = ['xgboost', 'lightgbm', 'catboost']
        best_params = {}
        
        for model_type in model_types:
            logger.info(f"Optimizing {model_type}...")
            params = self.hyperparameter_tuner.optimize_with_cv(
                model_type, X_train, y_train, n_trials
            )
            best_params[model_type] = params
            
        return best_params
    
    def _train_base_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame, y_val: pd.Series,
                          best_params: Dict[str, Dict[str, Any]]):
        """Train all base models."""
        # XGBoost
        xgb_params = best_params.get('xgboost', {})
        xgb_model = XGBoostModel(self.save_dir)
        xgb_model.build_model(**xgb_params)
        xgb_model.train(X_train, y_train, X_val, y_val)
        self.base_models['xgboost'] = xgb_model
        
        # LightGBM
        lgb_params = best_params.get('lightgbm', {})
        lgb_model = LightGBMModel(self.save_dir)
        lgb_model.build_model(**lgb_params)
        lgb_model.train(X_train, y_train, X_val, y_val)
        self.base_models['lightgbm'] = lgb_model
        
        # CatBoost
        cb_params = best_params.get('catboost', {})
        cb_model = CatBoostModel(self.save_dir)
        cb_model.build_model(**cb_params)
        cb_model.train(X_train, y_train, X_val, y_val)
        self.base_models['catboost'] = cb_model
        
        logger.info(f"Trained {len(self.base_models)} base models")
    
    def _evaluate_models_with_cv(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Evaluate all models using cross-validation."""
        cv_results = {}
        
        for name, model in self.base_models.items():
            results = self.cv_pipeline.evaluate_model(model, X_train, y_train)
            cv_results[name] = results
            
        return cv_results
    
    def _analyze_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Analyze feature importance across all models."""
        return self.feature_analyzer.analyze_all_models(self.base_models, feature_names)
    
    def _train_and_evaluate_ensembles(self, X_train: pd.DataFrame, y_train: pd.Series,
                                    X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Train and evaluate ensemble models."""
        self.ensemble_trainer = EnsembleTrainer(self.base_models)
        
        # Train ensembles
        ensembles = self.ensemble_trainer.train_all_ensembles(X_train, y_train)
        
        # Evaluate ensembles
        ensemble_results = self.ensemble_trainer.evaluate_ensembles(X_test, y_test)
        
        return ensemble_results
    
    def _evaluate_all_models_on_test(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """Evaluate all base models on test set."""
        test_results = {}
        
        for name, model in self.base_models.items():
            metrics = model.evaluate(X_test, y_test)
            test_results[name] = metrics
            
            logger.info(f"{name} test results:")
            logger.info(f"  F1: {metrics['f1_score']:.4f}")
            logger.info(f"  Precision: {metrics['precision']:.4f}")
            logger.info(f"  Recall: {metrics['recall']:.4f}")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
            
        return test_results
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results to file."""
        results_file = self.save_dir / f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Deep convert the results
        json_str = json.dumps(results, default=convert_numpy, indent=2)
        
        with open(results_file, 'w') as f:
            f.write(json_str)
            
        logger.info(f"Pipeline results saved to {results_file}")


def main():
    """Example usage of the MultiModelTrainer."""
    logger.info("Multi-Model Training & Optimization System initialized!")


if __name__ == "__main__":
    main()
