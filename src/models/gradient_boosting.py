"""
Gradient boosting models for fraud detection (XGBoost, LightGBM, CatBoost).
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import logging

# Import gradient boosting libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import GradientBoostingClassifier

from .base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XGBoostModel(BaseModel):
    """XGBoost classifier for fraud detection."""
    
    def __init__(self, save_dir: str = "models/saved"):
        super().__init__("XGBoost", save_dir)
        self.default_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
    
    def build_model(self, **kwargs) -> xgb.XGBClassifier:
        """Build XGBoost model with given parameters."""
        params = {**self.default_params, **kwargs}
        self.model = xgb.XGBClassifier(**params)
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train XGBoost model."""
        if self.model is None:
            self.build_model()
        
        logger.info(f"Training {self.model_name}...")
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False
        )
        
        self.is_trained = True
        
        # Evaluate on training set
        train_metrics = self.evaluate(X_train, y_train)
        self.training_metrics['train'] = train_metrics
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_metrics['validation'] = val_metrics
            logger.info(f"Validation F1-Score: {val_metrics['f1_score']:.4f}")
        
        logger.info(f"{self.model_name} training completed!")
        return self.training_metrics

class LightGBMModel(BaseModel):
    """LightGBM classifier for fraud detection."""
    
    def __init__(self, save_dir: str = "models/saved"):
        super().__init__("LightGBM", save_dir)
        self.default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    def build_model(self, **kwargs) -> lgb.LGBMClassifier:
        """Build LightGBM model with given parameters."""
        params = {**self.default_params, **kwargs}
        self.model = lgb.LGBMClassifier(**params)
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train LightGBM model."""
        if self.model is None:
            self.build_model()
        
        logger.info(f"Training {self.model_name}...")
        
        # Prepare evaluation set
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        self.is_trained = True
        
        # Evaluate on training set
        train_metrics = self.evaluate(X_train, y_train)
        self.training_metrics['train'] = train_metrics
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_metrics['validation'] = val_metrics
            logger.info(f"Validation F1-Score: {val_metrics['f1_score']:.4f}")
        
        logger.info(f"{self.model_name} training completed!")
        return self.training_metrics

class CatBoostModel(BaseModel):
    """CatBoost classifier for fraud detection."""
    
    def __init__(self, save_dir: str = "models/saved"):
        super().__init__("CatBoost", save_dir)
        self.default_params = {
            'objective': 'Logloss',
            'eval_metric': 'AUC',
            'depth': 6,
            'learning_rate': 0.1,
            'iterations': 100,
            'random_seed': 42,
            'verbose': False,
            'thread_count': -1
        }
    
    def build_model(self, **kwargs) -> cb.CatBoostClassifier:
        """Build CatBoost model with given parameters."""
        params = {**self.default_params, **kwargs}
        self.model = cb.CatBoostClassifier(**params)
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train CatBoost model."""
        if self.model is None:
            self.build_model()
        
        logger.info(f"Training {self.model_name}...")
        
        # Prepare evaluation set
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False
        )
        
        self.is_trained = True
        
        # Evaluate on training set
        train_metrics = self.evaluate(X_train, y_train)
        self.training_metrics['train'] = train_metrics
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_metrics['validation'] = val_metrics
            logger.info(f"Validation F1-Score: {val_metrics['f1_score']:.4f}")
        
        logger.info(f"{self.model_name} training completed!")
        return self.training_metrics

class SklearnGBModel(BaseModel):
    """Scikit-learn Gradient Boosting classifier for baseline."""
    
    def __init__(self, save_dir: str = "models/saved"):
        super().__init__("SklearnGB", save_dir)
        self.default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'random_state': 42
        }
    
    def build_model(self, **kwargs) -> GradientBoostingClassifier:
        """Build sklearn Gradient Boosting model with given parameters."""
        params = {**self.default_params, **kwargs}
        self.model = GradientBoostingClassifier(**params)
        return self.model
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: Optional[pd.DataFrame] = None,
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train sklearn Gradient Boosting model."""
        if self.model is None:
            self.build_model()
        
        logger.info(f"Training {self.model_name}...")
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate on training set
        train_metrics = self.evaluate(X_train, y_train)
        self.training_metrics['train'] = train_metrics
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            val_metrics = self.evaluate(X_val, y_val)
            self.training_metrics['validation'] = val_metrics
            logger.info(f"Validation F1-Score: {val_metrics['f1_score']:.4f}")
        
        logger.info(f"{self.model_name} training completed!")
        return self.training_metrics

def create_all_models(save_dir: str = "models/saved") -> Dict[str, BaseModel]:
    """Create all gradient boosting models."""
    models = {
        'xgboost': XGBoostModel(save_dir),
        'lightgbm': LightGBMModel(save_dir),
        'catboost': CatBoostModel(save_dir),
        'sklearn_gb': SklearnGBModel(save_dir)
    }
    
    logger.info(f"Created {len(models)} gradient boosting models")
    return models

def compare_models(models: Dict[str, BaseModel], 
                  X_test: pd.DataFrame, 
                  y_test: pd.Series) -> pd.DataFrame:
    """Compare performance of multiple models."""
    results = []
    
    for name, model in models.items():
        if model.is_trained:
            metrics = model.evaluate(X_test, y_test)
            metrics['model'] = name
            results.append(metrics)
    
    if not results:
        logger.warning("No trained models found for comparison")
        return pd.DataFrame()
    
    df_results = pd.DataFrame(results)
    df_results = df_results.set_index('model')
    
    # Sort by F1-score
    df_results = df_results.sort_values('f1_score', ascending=False)
    
    logger.info("Model comparison completed!")
    return df_results
