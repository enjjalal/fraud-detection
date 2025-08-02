"""
FastAPI application for fraud detection predictions.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import json
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for models
models = {}
scaler = None
feature_names = []

class TransactionInput(BaseModel):
    """Input schema for single transaction prediction."""
    Time: float = Field(..., description="Time elapsed since first transaction")
    V1: float = Field(..., description="PCA feature V1")
    V2: float = Field(..., description="PCA feature V2")
    V3: float = Field(..., description="PCA feature V3")
    V4: float = Field(..., description="PCA feature V4")
    V5: float = Field(..., description="PCA feature V5")
    V6: float = Field(..., description="PCA feature V6")
    V7: float = Field(..., description="PCA feature V7")
    V8: float = Field(..., description="PCA feature V8")
    V9: float = Field(..., description="PCA feature V9")
    V10: float = Field(..., description="PCA feature V10")
    V11: float = Field(..., description="PCA feature V11")
    V12: float = Field(..., description="PCA feature V12")
    V13: float = Field(..., description="PCA feature V13")
    V14: float = Field(..., description="PCA feature V14")
    V15: float = Field(..., description="PCA feature V15")
    V16: float = Field(..., description="PCA feature V16")
    V17: float = Field(..., description="PCA feature V17")
    V18: float = Field(..., description="PCA feature V18")
    V19: float = Field(..., description="PCA feature V19")
    V20: float = Field(..., description="PCA feature V20")
    V21: float = Field(..., description="PCA feature V21")
    V22: float = Field(..., description="PCA feature V22")
    V23: float = Field(..., description="PCA feature V23")
    V24: float = Field(..., description="PCA feature V24")
    V25: float = Field(..., description="PCA feature V25")
    V26: float = Field(..., description="PCA feature V26")
    V27: float = Field(..., description="PCA feature V27")
    V28: float = Field(..., description="PCA feature V28")
    Amount: float = Field(..., ge=0, description="Transaction amount")
    
    @validator('Amount')
    def validate_amount(cls, v):
        if v < 0:
            raise ValueError('Amount must be non-negative')
        return v

class BatchTransactionInput(BaseModel):
    """Input schema for batch predictions."""
    transactions: List[TransactionInput] = Field(..., max_items=1000)

class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    confidence: str
    timestamp: str

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]

class ModelInfo(BaseModel):
    """Model information response."""
    model_name: str
    version: str
    features: List[str]
    performance_metrics: Dict[str, float]
    last_trained: str

async def load_models():
    """Load trained models and preprocessing components."""
    global models, scaler, feature_names
    
    models_dir = Path("models/saved")
    
    try:
        # Load best model (try to find optimized versions first)
        model_files = {
            'xgboost': 'XGBoost_optimized.joblib',
            'lightgbm': 'LightGBM_optimized.joblib',
            'catboost': 'CatBoost_optimized.joblib'
        }
        
        # Fallback to regular models if optimized not found
        fallback_files = {
            'xgboost': 'XGBoost.joblib',
            'lightgbm': 'LightGBM.joblib',
            'catboost': 'CatBoost.joblib'
        }
        
        for model_name, filename in model_files.items():
            model_path = models_dir / filename
            fallback_path = models_dir / fallback_files[model_name]
            
            if model_path.exists():
                models[model_name] = joblib.load(model_path)
                logger.info(f"Loaded optimized {model_name} model")
            elif fallback_path.exists():
                models[model_name] = joblib.load(fallback_path)
                logger.info(f"Loaded {model_name} model")
        
        # Load scaler
        scaler_path = models_dir.parent.parent / "data/processed/standard_scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info("Loaded feature scaler")
        
        # Load feature names
        feature_path = models_dir.parent.parent / "data/processed/X_train.pkl"
        if feature_path.exists():
            sample_data = pd.read_pickle(feature_path)
            feature_names = list(sample_data.columns)
            logger.info(f"Loaded {len(feature_names)} feature names")
        
        if not models:
            logger.warning("No models loaded - using dummy model")
            # Create a dummy model for demonstration
            from sklearn.ensemble import RandomForestClassifier
            dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
            # Create dummy training data
            X_dummy = np.random.randn(100, 30)
            y_dummy = np.random.choice([0, 1], 100, p=[0.99, 0.01])
            dummy_model.fit(X_dummy, y_dummy)
            models['dummy'] = dummy_model
            feature_names = [f'feature_{i}' for i in range(30)]
            logger.info("Created dummy model for demonstration")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Fraud Detection API...")
    await load_models()
    logger.info("API startup completed")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Fraud Detection API...")

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Advanced Credit Card Fraud Detection using Machine Learning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def preprocess_transaction(transaction: TransactionInput) -> np.ndarray:
    """Preprocess a single transaction for prediction."""
    # Convert to DataFrame
    data = transaction.dict()
    df = pd.DataFrame([data])
    
    # Apply feature engineering (simplified version)
    if 'Amount' in df.columns:
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_zscore'] = (df['Amount'] - df['Amount'].mean()) / (df['Amount'].std() + 1e-8)
    
    if 'Time' in df.columns:
        df['Hour'] = (df['Time'] / 3600) % 24
        df['Day'] = (df['Time'] / (3600 * 24)) % 7
    
    # Add interaction features
    if 'V1' in df.columns and 'V2' in df.columns:
        df['V1_V2_interaction'] = df['V1'] * df['V2']
    
    # Select features that match training data
    if feature_names:
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0  # Fill missing features with 0
        
        df = df[feature_names]
    
    # Apply scaling if scaler is available
    if scaler is not None:
        try:
            scaled_data = scaler.transform(df)
            return scaled_data
        except Exception as e:
            logger.warning(f"Scaling failed: {e}, using raw data")
            return df.values
    
    return df.values

def get_ensemble_prediction(X: np.ndarray) -> tuple:
    """Get ensemble prediction from all loaded models."""
    if not models:
        return 0, 0.1  # Default safe prediction
    
    predictions = []
    probabilities = []
    
    for model_name, model in models.items():
        try:
            pred = model.predict(X)[0]
            
            if hasattr(model, 'predict_proba'):
                prob = model.predict_proba(X)[0, 1]
            else:
                prob = 0.5 if pred == 1 else 0.1
            
            predictions.append(pred)
            probabilities.append(prob)
            
        except Exception as e:
            logger.warning(f"Prediction failed for {model_name}: {e}")
            continue
    
    if not predictions:
        return 0, 0.1
    
    # Ensemble prediction (majority vote)
    final_prediction = int(np.mean(predictions) > 0.5)
    final_probability = np.mean(probabilities)
    
    return final_prediction, final_probability

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_status = {name: "loaded" for name in models.keys()}
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": model_status,
        "scaler_loaded": scaler is not None,
        "feature_count": len(feature_names)
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionInput):
    """Predict fraud for a single transaction."""
    try:
        # Preprocess transaction
        X = preprocess_transaction(transaction)
        
        # Get prediction
        prediction, probability = get_ensemble_prediction(X)
        
        # Calculate risk score and confidence
        risk_score = min(probability * 100, 100)
        
        if probability > 0.8:
            confidence = "high"
        elif probability > 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Generate transaction ID
        transaction_id = f"txn_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        return PredictionResponse(
            transaction_id=transaction_id,
            is_fraud=bool(prediction),
            fraud_probability=round(probability, 4),
            risk_score=round(risk_score, 2),
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(batch_input: BatchTransactionInput):
    """Predict fraud for multiple transactions."""
    try:
        predictions = []
        fraud_count = 0
        total_risk = 0
        
        for i, transaction in enumerate(batch_input.transactions):
            # Preprocess transaction
            X = preprocess_transaction(transaction)
            
            # Get prediction
            prediction, probability = get_ensemble_prediction(X)
            
            # Calculate metrics
            risk_score = min(probability * 100, 100)
            total_risk += risk_score
            
            if prediction:
                fraud_count += 1
            
            confidence = "high" if probability > 0.8 else "medium" if probability > 0.5 else "low"
            
            # Generate transaction ID
            transaction_id = f"batch_txn_{i+1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            predictions.append(PredictionResponse(
                transaction_id=transaction_id,
                is_fraud=bool(prediction),
                fraud_probability=round(probability, 4),
                risk_score=round(risk_score, 2),
                confidence=confidence,
                timestamp=datetime.now().isoformat()
            ))
        
        # Calculate summary statistics
        total_transactions = len(batch_input.transactions)
        fraud_rate = (fraud_count / total_transactions) * 100
        avg_risk_score = total_risk / total_transactions
        
        summary = {
            "total_transactions": total_transactions,
            "fraud_detected": fraud_count,
            "fraud_rate_percent": round(fraud_rate, 2),
            "average_risk_score": round(avg_risk_score, 2),
            "processing_time_ms": 0  # Could add actual timing
        }
        
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/model/info", response_model=Dict[str, Any])
async def get_model_info():
    """Get information about loaded models."""
    try:
        model_info = {}
        
        for model_name, model in models.items():
            info = {
                "name": model_name,
                "type": type(model).__name__,
                "features": len(feature_names),
                "loaded": True
            }
            
            # Try to get model parameters
            if hasattr(model, 'get_params'):
                info["parameters"] = model.get_params()
            
            model_info[model_name] = info
        
        return {
            "models": model_info,
            "feature_names": feature_names[:10],  # First 10 features
            "total_features": len(feature_names),
            "scaler_type": type(scaler).__name__ if scaler else None,
            "api_version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

@app.get("/model/performance")
async def get_model_performance():
    """Get model performance metrics."""
    try:
        # Try to load performance metrics from training
        metrics_path = Path("models/saved/training_report.json")
        
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                report = json.load(f)
            
            return {
                "training_date": report.get("timestamp"),
                "data_info": report.get("data_info"),
                "model_results": report.get("results", {}).get("individual_results", {}),
                "best_model": report.get("results", {}).get("best_model"),
                "ensemble_performance": report.get("results", {}).get("ensemble_results", {})
            }
        else:
            return {
                "message": "No performance metrics available",
                "suggestion": "Train models first using the training pipeline"
            }
            
    except Exception as e:
        logger.error(f"Performance metrics error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get performance metrics: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
