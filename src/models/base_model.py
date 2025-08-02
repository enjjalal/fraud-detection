"""
Base model class for fraud detection models.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all fraud detection models."""
    
    def __init__(self, model_name: str, save_dir: str = "models/saved"):
        self.model_name = model_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.is_trained = False
        self.feature_importance_ = None
        self.training_metrics = {}
        
    @abstractmethod
    def build_model(self, **kwargs) -> Any:
        """Build the model with given parameters."""
        pass
    
    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: Optional[pd.DataFrame] = None, 
              y_val: Optional[pd.Series] = None) -> Dict[str, float]:
        """Train the model."""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, use decision_function
            scores = self.model.decision_function(X)
            # Convert to probabilities using sigmoid
            proba = 1 / (1 + np.exp(-scores))
            return np.column_stack([1 - proba, proba])
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]  # Probability of positive class
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        return metrics
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """Get feature importance if available."""
        if not self.is_trained:
            return None
            
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                name='importance'
            )
        elif hasattr(self.model, 'coef_'):
            return pd.Series(
                np.abs(self.model.coef_[0]),
                name='importance'
            )
        else:
            return None
    
    def save_model(self, suffix: str = "") -> str:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_filename = f"{self.model_name}{suffix}.joblib"
        model_path = self.save_dir / model_filename
        
        # Save model
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'model_params': self.get_params() if hasattr(self, 'get_params') else {}
        }
        
        metadata_path = self.save_dir / f"{self.model_name}{suffix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return str(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load a trained model."""
        try:
            model_path = Path(model_path)
            
            # Load model
            self.model = joblib.load(model_path)
            self.is_trained = True
            
            # Load metadata if exists
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.training_metrics = metadata.get('training_metrics', {})
            
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        info = {
            'model_name': self.model_name,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics
        }
        
        if hasattr(self.model, 'get_params'):
            info['parameters'] = self.model.get_params()
        
        return info
    
    def print_evaluation_report(self, X: pd.DataFrame, y: pd.Series, 
                              dataset_name: str = "Dataset") -> None:
        """Print detailed evaluation report."""
        if not self.is_trained:
            print("❌ Model not trained yet")
            return
        
        print(f"\n{'='*50}")
        print(f"📊 {self.model_name} - {dataset_name} Evaluation")
        print(f"{'='*50}")
        
        # Get predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = self.evaluate(X, y)
        
        print(f"🎯 Accuracy:  {metrics['accuracy']:.4f}")
        print(f"🎯 Precision: {metrics['precision']:.4f}")
        print(f"🎯 Recall:    {metrics['recall']:.4f}")
        print(f"🎯 F1-Score:  {metrics['f1_score']:.4f}")
        print(f"🎯 ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        print(f"\n📈 Confusion Matrix:")
        print(f"    Predicted: [0, 1]")
        print(f"Actual 0: {cm[0]}")
        print(f"Actual 1: {cm[1]}")
        
        # Classification Report
        print(f"\n📋 Classification Report:")
        print(classification_report(y, y_pred, target_names=['Normal', 'Fraud']))

class ModelEnsemble:
    """Ensemble of multiple models."""
    
    def __init__(self, models: list, ensemble_method: str = 'voting'):
        self.models = models
        self.ensemble_method = ensemble_method
        self.weights = None
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> None:
        """Train all models in the ensemble."""
        logger.info(f"Training ensemble of {len(self.models)} models...")
        
        for model in self.models:
            logger.info(f"Training {model.model_name}...")
            model.train(X_train, y_train, X_val, y_val)
        
        # Calculate weights based on validation performance if available
        if X_val is not None and y_val is not None:
            self._calculate_weights(X_val, y_val)
    
    def _calculate_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """Calculate ensemble weights based on validation performance."""
        scores = []
        for model in self.models:
            metrics = model.evaluate(X_val, y_val)
            scores.append(metrics['f1_score'])  # Use F1-score for weighting
        
        # Normalize scores to get weights
        scores = np.array(scores)
        self.weights = scores / scores.sum()
        
        logger.info(f"Ensemble weights: {dict(zip([m.model_name for m in self.models], self.weights))}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if self.ensemble_method == 'voting':
            predictions = np.array([model.predict(X) for model in self.models])
            
            if self.weights is not None:
                # Weighted voting
                weighted_pred = np.average(predictions, axis=0, weights=self.weights)
                return (weighted_pred > 0.5).astype(int)
            else:
                # Simple majority voting
                return np.round(predictions.mean(axis=0)).astype(int)
        
        elif self.ensemble_method == 'average_proba':
            probabilities = np.array([model.predict_proba(X)[:, 1] for model in self.models])
            
            if self.weights is not None:
                avg_proba = np.average(probabilities, axis=0, weights=self.weights)
            else:
                avg_proba = probabilities.mean(axis=0)
            
            return (avg_proba > 0.5).astype(int)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ensemble probabilities."""
        probabilities = np.array([model.predict_proba(X) for model in self.models])
        
        if self.weights is not None:
            avg_proba = np.average(probabilities, axis=0, weights=self.weights)
        else:
            avg_proba = probabilities.mean(axis=0)
        
        return avg_proba
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate ensemble performance."""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_proba)
        }
        
        return metrics
