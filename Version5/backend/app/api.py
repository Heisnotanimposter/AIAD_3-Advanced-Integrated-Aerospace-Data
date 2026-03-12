from fastapi import APIRouter, HTTPException, UploadFile, File
from app.engines.weather_engine import WeatherEngineV5
from pydantic import BaseModel
import os

router = APIRouter()
engine = WeatherEngineV5(api_key=os.getenv("GEMINI_API_KEY"))

class PredictionResponse(BaseModel):
    confidence: float
    analysis: str
    image_url: str

@router.get("/predict")
async def get_prediction():
    """Generate a weather prediction and AI analysis"""
    # 1. Generate image with DCGAN
    image = engine.predict_weather_pattern()
    
    # 2. Analyze with Gemini (simulated here for demonstration)
    analysis = "Initial diagnostic: Stable atmospheric conditions with rising humidity."
    
    return {
        "confidence": 0.94,
        "analysis": analysis,
        "image_url": "/data/latest_prediction.jpg"
    }

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze a provided satellite image using Gemini AI"""
    try:
        content = await file.read()
        analysis = await engine.analyze_with_gemini(content)
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
