from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
from app.engines.weather_engine_v6 import WeatherEngineV6

router = APIRouter()
engine = WeatherEngineV6()

class PredictionRequest(BaseModel):
    forecast_horizon_hours: Optional[int] = 24
    format_type: Optional[str] = "json"

@router.get("/nations")
async def get_nations():
    """List all 28 integrated popular worldwide nations with dataset metadata."""
    try:
        nations = engine.get_integrated_nations()
        return {
            "status": "success",
            "count": len(nations),
            "nations": nations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nations/{country_code}/dataset")
async def get_nation_dataset(country_code: str, format_type: str = Query("json", description="json or csv")):
    """Fetch or download raw dataset records for a specified nation in JSON or CSV format."""
    try:
        records = engine.load_nation_records(country_code, format_type=format_type)
        return {
            "country_code": country_code.upper(),
            "format": format_type,
            "record_count": len(records),
            "data": records
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/nations/{country_code}/detect")
async def detect_cloud_form(country_code: str):
    """Run cloud form detection on historical/latest satellite time series data for a nation."""
    try:
        records = engine.load_nation_records(country_code, format_type="json")
        detection = engine.detect_cloud_formation(records[-1])
        return {
            "status": "success",
            "detection": detection
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/nations/{country_code}/predict")
async def predict_time_series(country_code: str, req: Optional[PredictionRequest] = None):
    """Run time-series prediction (+24h to +72h) for cloud cover, cloud form transitions, and satellite reflectance."""
    try:
        hours = req.forecast_horizon_hours if req and req.forecast_horizon_hours else 24
        forecast = engine.predict_time_series(country_code, hours=hours)
        records = engine.load_nation_records(country_code, format_type="json")
        detection = engine.detect_cloud_formation(records[-1])
        
        reasoning = await engine.generate_gemini_reasoning(
            forecast["country_name"], detection, forecast
        )
        
        return {
            "status": "success",
            "country_code": country_code.upper(),
            "latest_detection": detection,
            "forecast": forecast,
            "ai_reasoning": reasoning
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/batch-analysis")
async def run_batch_analysis():
    """Execute batch cloud detection and prediction across all 28 integrated popular nations."""
    try:
        nations = engine.get_integrated_nations()
        results = []
        for n in nations:
            code = n["code"]
            records = engine.load_nation_records(code, format_type="json")
            detection = engine.detect_cloud_formation(records[-1])
            forecast = engine.predict_time_series(code, hours=24)
            results.append({
                "code": code,
                "name": n["name"],
                "latitude": n["latitude"],
                "longitude": n["longitude"],
                "climate_zone": n["climate_zone"],
                "latest_cloud_cover_pct": detection["cloud_cover_pct"],
                "latest_detected_form": detection["detected_cloud_form"],
                "24h_predicted_cloud_cover_pct": forecast["predictions"][-1]["predicted_cloud_cover_pct"],
                "24h_predicted_form": forecast["predictions"][-1]["predicted_cloud_form"],
                "confidence": detection["confidence"]
            })
        return {
            "status": "success",
            "processed_nations": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
