"""
Batch processing endpoints for the Fraud Detection API.

This module provides:
- Async batch prediction processing
- Background task management
- Batch result storage and retrieval
- Progress tracking for large batches
"""

import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
from fastapi import HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# In-memory storage for batch jobs (in production, use Redis or database)
batch_jobs = {}
batch_results = {}

class BatchJob(BaseModel):
    """Batch job model."""
    job_id: str
    status: str  # pending, processing, completed, failed
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_transactions: int
    processed_transactions: int = 0
    progress_percent: float = 0.0
    estimated_completion: Optional[datetime] = None
    error_message: Optional[str] = None

class BatchJobResponse(BaseModel):
    """Batch job creation response."""
    job_id: str
    status: str
    message: str
    estimated_processing_time_minutes: float
    check_status_url: str

class BatchStatusResponse(BaseModel):
    """Batch job status response."""
    job_id: str
    status: str
    progress_percent: float
    processed_transactions: int
    total_transactions: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_completion: Optional[str] = None
    error_message: Optional[str] = None
    results_available: bool = False
    download_url: Optional[str] = None

class BatchResultSummary(BaseModel):
    """Batch processing results summary."""
    job_id: str
    total_transactions: int
    fraud_detected: int
    fraud_rate_percent: float
    average_fraud_probability: float
    average_risk_score: float
    high_risk_transactions: int
    processing_time_seconds: float
    completed_at: str

class BatchProcessor:
    """Handles batch processing of fraud predictions."""
    
    def __init__(self, models: Dict, scaler=None, feature_names: List[str] = None):
        self.models = models
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.max_concurrent_jobs = 3
        self.active_jobs = set()
    
    async def create_batch_job(self, transactions: List[Dict], background_tasks: BackgroundTasks) -> BatchJobResponse:
        """Create a new batch processing job."""
        job_id = str(uuid.uuid4())
        total_transactions = len(transactions)
        
        # Estimate processing time (roughly 10ms per transaction)
        estimated_time_minutes = (total_transactions * 0.01) / 60
        
        # Create job record
        job = BatchJob(
            job_id=job_id,
            status="pending",
            created_at=datetime.now(),
            total_transactions=total_transactions
        )
        
        batch_jobs[job_id] = job
        
        # Start processing in background
        background_tasks.add_task(
            self.process_batch,
            job_id,
            transactions
        )
        
        return BatchJobResponse(
            job_id=job_id,
            status="pending",
            message=f"Batch job created with {total_transactions} transactions",
            estimated_processing_time_minutes=round(estimated_time_minutes, 2),
            check_status_url=f"/batch/status/{job_id}"
        )
    
    async def process_batch(self, job_id: str, transactions: List[Dict]):
        """Process batch of transactions."""
        try:
            if len(self.active_jobs) >= self.max_concurrent_jobs:
                batch_jobs[job_id].status = "queued"
                # Wait for slot to become available
                while len(self.active_jobs) >= self.max_concurrent_jobs:
                    await asyncio.sleep(1)
            
            self.active_jobs.add(job_id)
            job = batch_jobs[job_id]
            
            # Update job status
            job.status = "processing"
            job.started_at = datetime.now()
            
            logger.info(f"Starting batch processing for job {job_id}")
            
            results = []
            fraud_count = 0
            total_probability = 0.0
            total_risk_score = 0.0
            high_risk_count = 0
            
            start_time = datetime.now()
            
            for i, transaction_data in enumerate(transactions):
                try:
                    # Process single transaction
                    prediction_result = await self._process_single_transaction(
                        transaction_data, f"{job_id}_{i}"
                    )
                    
                    results.append(prediction_result)
                    
                    # Update statistics
                    if prediction_result['is_fraud']:
                        fraud_count += 1
                    
                    total_probability += prediction_result['fraud_probability']
                    total_risk_score += prediction_result['risk_score']
                    
                    if prediction_result['risk_score'] > 80:
                        high_risk_count += 1
                    
                    # Update progress
                    job.processed_transactions = i + 1
                    job.progress_percent = (i + 1) / len(transactions) * 100
                    
                    # Estimate completion time
                    if i > 0:
                        elapsed = (datetime.now() - start_time).total_seconds()
                        avg_time_per_transaction = elapsed / (i + 1)
                        remaining_transactions = len(transactions) - (i + 1)
                        estimated_remaining_seconds = remaining_transactions * avg_time_per_transaction
                        job.estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
                    
                    # Small delay to prevent overwhelming the system
                    if i % 100 == 0:
                        await asyncio.sleep(0.01)
                        
                except Exception as e:
                    logger.error(f"Error processing transaction {i} in job {job_id}: {e}")
                    # Continue with next transaction
                    continue
            
            # Calculate final statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            total_transactions = len(transactions)
            
            summary = BatchResultSummary(
                job_id=job_id,
                total_transactions=total_transactions,
                fraud_detected=fraud_count,
                fraud_rate_percent=round((fraud_count / total_transactions) * 100, 2),
                average_fraud_probability=round(total_probability / total_transactions, 4),
                average_risk_score=round(total_risk_score / total_transactions, 2),
                high_risk_transactions=high_risk_count,
                processing_time_seconds=round(processing_time, 2),
                completed_at=datetime.now().isoformat()
            )
            
            # Store results
            batch_results[job_id] = {
                "summary": summary,
                "predictions": results
            }
            
            # Update job status
            job.status = "completed"
            job.completed_at = datetime.now()
            job.progress_percent = 100.0
            
            logger.info(f"Batch processing completed for job {job_id}")
            
        except Exception as e:
            logger.error(f"Batch processing failed for job {job_id}: {e}")
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now()
            
        finally:
            self.active_jobs.discard(job_id)
    
    async def _process_single_transaction(self, transaction_data: Dict, transaction_id: str) -> Dict:
        """Process a single transaction."""
        try:
            # Convert transaction data to array
            feature_values = [
                transaction_data.get('Time', 0),
                *[transaction_data.get(f'V{i}', 0) for i in range(1, 29)],
                transaction_data.get('Amount', 0)
            ]
            
            X = np.array(feature_values).reshape(1, -1)
            
            # Scale if scaler available
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Get ensemble prediction
            predictions = []
            probabilities = []
            
            for model_name, model in self.models.items():
                try:
                    pred = model.predict(X)[0]
                    prob = model.predict_proba(X)[0][1] if hasattr(model, 'predict_proba') else pred
                    predictions.append(pred)
                    probabilities.append(prob)
                except Exception as e:
                    logger.warning(f"Model {model_name} prediction failed: {e}")
            
            if not predictions:
                raise Exception("All model predictions failed")
            
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
            
            return {
                "transaction_id": transaction_id,
                "is_fraud": bool(final_prediction),
                "fraud_probability": round(final_probability, 4),
                "risk_score": round(risk_score, 2),
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing transaction {transaction_id}: {e}")
            return {
                "transaction_id": transaction_id,
                "is_fraud": False,
                "fraud_probability": 0.0,
                "risk_score": 0.0,
                "confidence": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_job_status(self, job_id: str) -> Optional[BatchStatusResponse]:
        """Get status of a batch job."""
        if job_id not in batch_jobs:
            return None
        
        job = batch_jobs[job_id]
        results_available = job_id in batch_results and job.status == "completed"
        
        return BatchStatusResponse(
            job_id=job_id,
            status=job.status,
            progress_percent=round(job.progress_percent, 2),
            processed_transactions=job.processed_transactions,
            total_transactions=job.total_transactions,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            estimated_completion=job.estimated_completion.isoformat() if job.estimated_completion else None,
            error_message=job.error_message,
            results_available=results_available,
            download_url=f"/batch/results/{job_id}" if results_available else None
        )
    
    def get_job_results(self, job_id: str) -> Optional[Dict]:
        """Get results of a completed batch job."""
        return batch_results.get(job_id)
    
    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old batch jobs and results."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        jobs_to_remove = []
        for job_id, job in batch_jobs.items():
            if job.created_at < cutoff_time:
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            batch_jobs.pop(job_id, None)
            batch_results.pop(job_id, None)
            logger.info(f"Cleaned up old batch job {job_id}")


# Global batch processor instance (will be initialized with models)
batch_processor = None


def initialize_batch_processor(models: Dict, scaler=None, feature_names: List[str] = None):
    """Initialize the global batch processor."""
    global batch_processor
    batch_processor = BatchProcessor(models, scaler, feature_names)


def get_batch_processor() -> BatchProcessor:
    """Get the batch processor instance."""
    if batch_processor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Batch processor not initialized"
        )
    return batch_processor
