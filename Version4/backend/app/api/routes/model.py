"""
Model management routes for DCGAN model operations
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
import random

from app.services.model_service import ModelService

router = APIRouter()
model_service = ModelService()

@router.get("/stats")
async def get_model_stats():
    """Get model statistics and information"""
    try:
        stats = await model_service.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_model_performance(timeRange: str = "7d"):
    """Get model performance metrics"""
    try:
        performance = await model_service.get_performance(timeRange)
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrain")
async def retrain_model(
    background_tasks: BackgroundTasks,
    epochs: int = 30,
    learning_rate: float = 0.0001
):
    """Retrain the DCGAN model"""
    try:
        background_tasks.add_task(model_service.retrain_async, epochs, learning_rate)
        return {
            "message": "Model retraining started",
            "epochs": epochs,
            "learning_rate": learning_rate,
            "estimated_time": f"{epochs * 2} minutes"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/training-history")
async def get_training_history():
    """Get model training history"""
    try:
        history = await model_service.get_training_history()
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for model service"""
    try:
        status = await model_service.health_check()
        return {"service": "model", "status": status}
    except Exception as e:
        return {"service": "model", "status": "unhealthy", "error": str(e)}
