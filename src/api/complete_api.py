"""
Complete Production-Ready FastAPI for Fraud Detection.

This module provides a comprehensive production-ready API compatible with the existing codebase:
- Async endpoints with rate limiting
- Input validation and error handling
- Health checks and monitoring
- Batch processing capabilities
- Integration with existing models
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks
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
from datetime import datetime, timedelta
import asyncio
import time
import uuid
import psutil
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
models = {}
scaler = None
feature_names = []
api_metrics = {
    'requests_total': 0,
    'requests_success': 0,
    'requests_error': 0,
    'avg_response_time': 0.0,
    'start_time': datetime.now()
}

# Rate limiting
rate_limit_storage = defaultdict(lambda: deque())
rate_limit_lock = threading.Lock()

class RateLimiter:
    """Simple rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds
        
        with rate_limit_lock:
            # Clean old requests
            while (rate_limit_storage[client_id] and 
                   rate_limit_storage[client_id][0] < window_start):
                rate_limit_storage[client_id].popleft()
            
            # Check limit
            if len(rate_limit_storage[client_id]) >= self.max_requests:
                return False
            
            # Add current request
            rate_limit_storage[client_id].append(now)
            return True

rate_limiter = RateLimiter()

# Pydantic models
class TransactionInput(BaseModel):
    """Input schema for transaction prediction."""
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

class BatchInput(BaseModel):
    """Batch prediction input."""
    transactions: List[TransactionInput] = Field(..., max_items=1000)

class PredictionResponse(BaseModel):
    """Prediction response."""
    transaction_id: str
    is_fraud: bool
    fraud_probability: float
    risk_score: float
    confidence: str
    timestamp: str
    processing_time_ms: float

class BatchResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]
    processing_time_ms: float

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    models_loaded: int
    system_info: Dict[str, Any]

async def load_models():
    """Load trained models compatible with existing codebase."""
    global models, scaler, feature_names
    
    try:
        models_dir = Path("models/saved")
        
        # Try to load models from multi-model trainer
        model_files = {
            'xgboost': ['XGBoost_optimized.joblib', 'XGBoost.joblib'],
            'lightgbm': ['LightGBM_optimized.joblib', 'LightGBM.joblib'],
            'catboost': ['CatBoost_optimized.joblib', 'CatBoost.joblib']
        }
        
        for model_name, file_options in model_files.items():
            for filename in file_options:
                model_path = models_dir / filename
                if model_path.exists():
                    models[model_name] = joblib.load(model_path)
                    logger.info(f"Loaded {model_name} from {filename}")
                    break
        
        # Load scaler if available
        scaler_path = models_dir / "scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler")
        
        # Set feature names compatible with existing code
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        if not models:
            logger.warning("No models loaded - API will use fallback predictions")
        else:
            logger.info(f"Successfully loaded {len(models)} models")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Fraud Detection API...")
    await load_models()
    logger.info("API startup complete")
    yield
    logger.info("Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Production-Ready Credit Card Fraud Detection API",
    version="2.0.0",
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

# Dependency functions
async def get_client_id(request: Request) -> str:
    """Get client identifier."""
    return request.client.host if request.client else "unknown"

async def check_rate_limit(client_id: str = Depends(get_client_id)):
    """Check rate limit."""
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )

# Middleware for metrics
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect metrics and handle errors."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        request.state.request_id = request_id
        api_metrics['requests_total'] += 1
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        if response.status_code < 400:
            api_metrics['requests_success'] += 1
        else:
            api_metrics['requests_error'] += 1
        
        # Update average response time
        total = api_metrics['requests_total']
        current_avg = api_metrics['avg_response_time']
        api_metrics['avg_response_time'] = (current_avg * (total - 1) + process_time) / total
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
        
    except Exception as e:
        api_metrics['requests_error'] += 1
        logger.error(f"Request {request_id} failed: {e}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "request_id": request_id
            }
        )

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Fraud Detection API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = (datetime.now() - api_metrics['start_time']).total_seconds()
    
    system_info = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0
    }
    
    return HealthResponse(
        status="healthy" if models else "degraded",
        timestamp=datetime.now().isoformat(),
        version="2.0.0",
        uptime_seconds=uptime,
        models_loaded=len(models),
        system_info=system_info
    )

@app.get("/metrics")
async def get_metrics():
    """Get API metrics."""
    return {
        "api_metrics": api_metrics,
        "rate_limit_stats": {
            "active_clients": len(rate_limit_storage)
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    transaction: TransactionInput,
    request: Request,
    _: None = Depends(check_rate_limit)
):
    """Predict fraud for a single transaction."""
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    try:
        if not models:
            # Fallback prediction based on amount and features
            fallback_prob = min(transaction.Amount / 10000, 0.9) if transaction.Amount > 1000 else 0.1
            return PredictionResponse(
                transaction_id=f"txn_{request_id[:8]}",
                is_fraud=fallback_prob > 0.5,
                fraud_probability=round(fallback_prob, 4),
                risk_score=round(fallback_prob * 100, 2),
                confidence="low",
                timestamp=datetime.now().isoformat(),
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )
        
        # Prepare transaction data
        transaction_data = [
            transaction.Time, transaction.V1, transaction.V2, transaction.V3,
            transaction.V4, transaction.V5, transaction.V6, transaction.V7,
            transaction.V8, transaction.V9, transaction.V10, transaction.V11,
            transaction.V12, transaction.V13, transaction.V14, transaction.V15,
            transaction.V16, transaction.V17, transaction.V18, transaction.V19,
            transaction.V20, transaction.V21, transaction.V22, transaction.V23,
            transaction.V24, transaction.V25, transaction.V26, transaction.V27,
            transaction.V28, transaction.Amount
        ]
        
        X = np.array(transaction_data).reshape(1, -1)
        
        # Scale if available
        if scaler:
            X = scaler.transform(X)
        
        # Get ensemble prediction
        predictions = []
        probabilities = []
        
        for model_name, model in models.items():
            try:
                pred = model.predict(X)[0]
                prob = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else pred
                predictions.append(pred)
                probabilities.append(prob)
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
        
        if not predictions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="All model predictions failed"
            )
        
        # Ensemble result
        final_probability = np.mean(probabilities)
        final_prediction = final_probability > 0.5
        risk_score = min(final_probability * 100, 100)
        
        confidence = "high" if final_probability > 0.8 else "medium" if final_probability > 0.5 else "low"
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            transaction_id=f"txn_{request_id[:8]}",
            is_fraud=bool(final_prediction),
            fraud_probability=round(final_probability, 4),
            risk_score=round(risk_score, 2),
            confidence=confidence,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict/batch", response_model=BatchResponse)
async def predict_batch(
    batch_input: BatchInput,
    request: Request,
    _: None = Depends(check_rate_limit)
):
    """Predict fraud for multiple transactions."""
    start_time = time.time()
    
    try:
        predictions = []
        fraud_count = 0
        total_probability = 0.0
        total_risk = 0.0
        
        for i, transaction in enumerate(batch_input.transactions):
            # Create a mock request for individual prediction
            mock_request = type('MockRequest', (), {'state': type('State', (), {'request_id': f"batch_{i}"})()})()
            
            try:
                prediction = await predict_fraud(transaction, mock_request)
                predictions.append(prediction)
                
                if prediction.is_fraud:
                    fraud_count += 1
                
                total_probability += prediction.fraud_probability
                total_risk += prediction.risk_score
                
            except Exception as e:
                logger.error(f"Batch item {i} failed: {e}")
                # Add error prediction
                predictions.append(PredictionResponse(
                    transaction_id=f"batch_error_{i}",
                    is_fraud=False,
                    fraud_probability=0.0,
                    risk_score=0.0,
                    confidence="error",
                    timestamp=datetime.now().isoformat(),
                    processing_time_ms=0.0
                ))
        
        total_transactions = len(batch_input.transactions)
        processing_time = (time.time() - start_time) * 1000
        
        summary = {
            "total_transactions": total_transactions,
            "fraud_detected": fraud_count,
            "fraud_rate_percent": round((fraud_count / total_transactions) * 100, 2),
            "average_fraud_probability": round(total_probability / total_transactions, 4),
            "average_risk_score": round(total_risk / total_transactions, 2)
        }
        
        return BatchResponse(
            predictions=predictions,
            summary=summary,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.get("/models/info")
async def get_model_info():
    """Get information about loaded models."""
    try:
        model_info = {}
        
        for model_name, model in models.items():
            info = {
                "name": model_name,
                "type": type(model).__name__,
                "loaded": True
            }
            model_info[model_name] = info
        
        return {
            "models": model_info,
            "total_models": len(models),
            "scaler_loaded": scaler is not None,
            "feature_count": len(feature_names),
            "api_version": "2.0.0"
        }
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "complete_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
