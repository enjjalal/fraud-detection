"""
Production-Ready FastAPI Application for Fraud Detection.

This module provides a comprehensive production-ready API with:
- Async endpoints for high performance
- Input validation with Pydantic
- Rate limiting and security
- Health checks and monitoring
- Comprehensive error handling
- API documentation and versioning
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import asyncio
import time
import hashlib
import psutil
import aiofiles
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import threading
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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

# Rate limiting storage
rate_limit_storage = defaultdict(lambda: deque())
rate_limit_lock = threading.Lock()

# Security
security = HTTPBearer(auto_error=False)

class RateLimiter:
    """Rate limiting implementation."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
    
    async def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed for client."""
        now = time.time()
        window_start = now - self.window_seconds
        
        with rate_limit_lock:
            # Clean old requests
            while (rate_limit_storage[client_id] and 
                   rate_limit_storage[client_id][0] < window_start):
                rate_limit_storage[client_id].popleft()
            
            # Check if under limit
            if len(rate_limit_storage[client_id]) >= self.max_requests:
                return False
            
            # Add current request
            rate_limit_storage[client_id].append(now)
            return True

# Rate limiter instance
rate_limiter = RateLimiter(max_requests=100, window_seconds=60)

# Pydantic models
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
    processing_time_ms: float

class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    summary: Dict[str, Any]
    batch_id: str
    processing_time_ms: float

class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    models_loaded: int
    system_info: Dict[str, Any]

class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    message: str
    timestamp: str
    request_id: str

async def load_models():
    """Load trained models and preprocessing components."""
    global models, scaler, feature_names
    
    try:
        models_dir = Path("models/saved")
        
        # Load models
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
        
        # Load scaler
        scaler_path = models_dir / "scaler.joblib"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            logger.info("Loaded scaler")
        
        # Set feature names
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        
        logger.info(f"Successfully loaded {len(models)} models")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Fraud Detection API...")
    await load_models()
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")

# Create FastAPI app
app = FastAPI(
    title="Fraud Detection API",
    description="Production-Ready Credit Card Fraud Detection API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1", "yourdomain.com"]
)

# Dependency functions
async def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting."""
    return request.client.host if request.client else "unknown"

async def check_rate_limit(client_id: str = Depends(get_client_id)):
    """Check rate limit for client."""
    if not await rate_limiter.is_allowed(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (optional)."""
    # For demo purposes, accept any token or no token
    # In production, implement proper JWT verification
    return True

# Middleware for metrics and error handling
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware for collecting metrics and error handling."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Update metrics
        api_metrics['requests_total'] += 1
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        process_time = time.time() - start_time
        
        # Update success metrics
        if response.status_code < 400:
            api_metrics['requests_success'] += 1
        else:
            api_metrics['requests_error'] += 1
        
        # Update average response time
        total_requests = api_metrics['requests_total']
        current_avg = api_metrics['avg_response_time']
        api_metrics['avg_response_time'] = (
            (current_avg * (total_requests - 1) + process_time) / total_requests
        )
        
        # Add headers
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
@app.get("/", response_model=Dict[str, str])
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
    """Comprehensive health check endpoint."""
    uptime = (datetime.now() - api_metrics['start_time']).total_seconds()
    
    # System information
    system_info = {
        "cpu_percent": psutil.cpu_percent(),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}"
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
            "active_clients": len(rate_limit_storage),
            "max_requests_per_minute": rate_limiter.max_requests
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    transaction: TransactionInput,
    background_tasks: BackgroundTasks,
    request: Request,
    _: bool = Depends(check_rate_limit),
    __: bool = Depends(verify_token)
):
    """Predict fraud for a single transaction."""
    start_time = time.time()
    request_id = getattr(request.state, 'request_id', str(uuid.uuid4()))
    
    try:
        if not models:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Models not loaded"
            )
        
        # Preprocess transaction
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
        
        # Scale if scaler available
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
                logger.warning(f"Model {model_name} prediction failed: {e}")
        
        if not predictions:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="All model predictions failed"
            )
        
        # Ensemble prediction
        final_probability = np.mean(probabilities)
        final_prediction = final_probability > 0.5
        risk_score = min(final_probability * 100, 100)
        
        # Determine confidence
        if final_probability > 0.8:
            confidence = "high"
        elif final_probability > 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        processing_time = (time.time() - start_time) * 1000
        
        # Log prediction (background task)
        background_tasks.add_task(
            log_prediction,
            request_id,
            final_prediction,
            final_probability,
            processing_time
        )
        
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
        logger.error(f"Prediction error for request {request_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

async def log_prediction(request_id: str, prediction: bool, probability: float, processing_time: float):
    """Background task to log predictions."""
    try:
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "prediction": prediction,
            "probability": probability,
            "processing_time_ms": processing_time
        }
        
        # In production, save to database or log file
        logger.info(f"Prediction logged: {log_entry}")
        
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "production_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )
