"""
Prediction routes for DCGAN weather predictions
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, List, Dict, Any
from datetime import datetime
import asyncio

from app.services.prediction_service import PredictionService

router = APIRouter()
prediction_service = PredictionService()

@router.get("/")
async def get_predictions(
    time_horizon: int = Query(default=24),
    region: str = Query(default="global"),
    type: str = Query(default="hourly")
):
    """Get weather predictions with filters"""
    try:
        data = await prediction_service.get_predictions(time_horizon, region, type)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recent")
async def get_recent_predictions(limit: int = Query(default=10)):
    """Get recent predictions"""
    try:
        predictions = await prediction_service.get_recent_predictions(limit)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate")
async def generate_prediction(
    background_tasks: BackgroundTasks,
    time_horizon: int = Query(default=24),
    region: str = Query(default="global"),
    type: str = Query(default="hourly")
):
    """Generate new weather prediction using DCGAN"""
    try:
        # Start background task for generation
        background_tasks.add_task(
            prediction_service.generate_prediction_async,
            time_horizon,
            region,
            type
        )
        
        return {
            "message": "Prediction generation started",
            "time_horizon": time_horizon,
            "region": region,
            "type": type,
            "status": "processing",
            "estimated_time": "30 seconds"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{prediction_id}")
async def get_prediction(prediction_id: int):
    """Get specific prediction by ID"""
    try:
        prediction = await prediction_service.get_prediction_by_id(prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{prediction_id}/confidence")
async def get_confidence_scores(prediction_id: int):
    """Get confidence scores for a prediction"""
    try:
        scores = await prediction_service.get_confidence_scores(prediction_id)
        return scores
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{prediction_id}")
async def delete_prediction(prediction_id: int):
    """Delete a prediction"""
    try:
        success = await prediction_service.delete_prediction(prediction_id)
        if not success:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return {"message": "Prediction deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_prediction_stats():
    """Get prediction statistics"""
    try:
        stats = await prediction_service.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for prediction service"""
    try:
        status = await prediction_service.health_check()
        return {"service": "predictions", "status": status}
    except Exception as e:
        return {"service": "predictions", "status": "unhealthy", "error": str(e)}
