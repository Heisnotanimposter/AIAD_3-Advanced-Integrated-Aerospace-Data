from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import asyncio
import os
from datetime import datetime
from typing import List, Optional

from app.api.routes import satellite, predictions, analytics, model
from app.core.config import settings
from app.core.database import init_db
from app.services.satellite_service import SatelliteService
from app.services.prediction_service import PredictionService
from app.services.analytics_service import AnalyticsService

app = FastAPI(
    title="Satellite Data Analysis API",
    description="Advanced API for satellite imagery processing and weather prediction using DCGAN",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_HOSTS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for generated images
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(satellite.router, prefix="/api/satellite", tags=["satellite"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])
app.include_router(model.router, prefix="/api/model", tags=["model"])

# Initialize services
satellite_service = SatelliteService()
prediction_service = PredictionService()
analytics_service = AnalyticsService()

@app.on_event("startup")
async def startup_event():
    """Initialize database and services on startup"""
    await init_db()
    print("🚀 Satellite Data Analysis API v4.0 started successfully")
    print("📊 Dashboard: http://localhost:8000/docs")
    print("🛰️ Frontend: http://localhost:3000")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Satellite Data Analysis API v4.0",
        "status": "active",
        "version": "4.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "satellite": "/api/satellite",
            "predictions": "/api/predictions",
            "analytics": "/api/analytics",
            "model": "/api/model"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "satellite": await satellite_service.health_check(),
            "predictions": await prediction_service.health_check(),
            "analytics": await analytics_service.health_check()
        }
    }

@app.get("/api/stats")
async def get_stats():
    """Get system statistics"""
    try:
        satellite_stats = await satellite_service.get_stats()
        prediction_stats = await prediction_service.get_stats()
        analytics_stats = await analytics_service.get_stats()
        
        return {
            "satellite": satellite_stats,
            "predictions": prediction_stats,
            "analytics": analytics_stats,
            "system": {
                "uptime": "2d 14h 32m",
                "version": "4.0.0",
                "last_update": datetime.now().isoformat()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-sample-data")
async def generate_sample_data(background_tasks: BackgroundTasks):
    """Generate sample data for testing"""
    background_tasks.add_task(generate_sample_data_task)
    return {"message": "Sample data generation started"}

async def generate_sample_data_task():
    """Background task to generate sample data"""
    try:
        await satellite_service.generate_sample_data()
        await prediction_service.generate_sample_predictions()
        await analytics_service.generate_sample_analytics()
        print("✅ Sample data generated successfully")
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
