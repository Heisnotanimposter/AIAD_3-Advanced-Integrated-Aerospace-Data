"""
Analytics routes for performance metrics and data analysis
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
import random
from datetime import datetime, timedelta

from app.services.analytics_service import AnalyticsService

router = APIRouter()
analytics_service = AnalyticsService()

@router.get("/")
async def get_analytics(
    timeRange: str = Query(default="7d"),
    metric: str = Query(default="all")
):
    """Get comprehensive analytics data"""
    try:
        data = await analytics_service.get_analytics(timeRange, metric)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance")
async def get_performance_metrics(timeRange: str = Query(default="7d")):
    """Get detailed performance metrics"""
    try:
        metrics = await analytics_service.get_performance_metrics(timeRange)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/regional")
async def get_regional_analytics():
    """Get regional analytics data"""
    try:
        data = await analytics_service.get_regional_analytics()
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/usage")
async def get_usage_stats(timeRange: str = Query(default="7d")):
    """Get usage statistics"""
    try:
        stats = await analytics_service.get_usage_stats(timeRange)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export")
async def export_analytics(
    timeRange: str = Query(default="7d"),
    format: str = Query(default="csv")
):
    """Export analytics data"""
    try:
        data = await analytics_service.export_analytics(timeRange, format)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/realtime")
async def get_realtime_metrics():
    """Get real-time metrics"""
    try:
        metrics = await analytics_service.get_realtime_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model-comparison")
async def get_model_comparison():
    """Get model comparison data"""
    try:
        comparison = await analytics_service.get_model_comparison()
        return comparison
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check for analytics service"""
    try:
        status = await analytics_service.health_check()
        return {"service": "analytics", "status": status}
    except Exception as e:
        return {"service": "analytics", "status": "unhealthy", "error": str(e)}
