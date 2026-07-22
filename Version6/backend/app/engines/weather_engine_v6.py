import os
import json
import csv
import math
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
import google.generativeai as genai

class WeatherEngineV6:
    """
    Version 6 Weather Engine: Global Multi-Nation Cloud Detection,
    Time-Series Prediction, and Multimodal Gemini AI Reasoning.
    """
    
    def __init__(self, data_dir: Optional[str] = None, api_key: Optional[str] = None):
        if data_dir is None:
            # Locate data directory
            possible_dirs = [
                "/Users/seungwonlee/AIAD_weather2/Version6/data/nations",
                "../data/nations",
                "./data/nations"
            ]
            self.data_dir = next((d for d in possible_dirs if os.path.exists(d)), possible_dirs[0])
        else:
            self.data_dir = data_dir
            
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
            except Exception:
                self.gemini_model = None
        else:
            self.gemini_model = None

    def get_integrated_nations(self) -> List[Dict]:
        """Return list of all integrated 28 popular nations."""
        manifest_path = os.path.join(self.data_dir, "nations_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def load_nation_records(self, country_code: str, format_type: str = "json") -> List[Dict]:
        """Load CSV or JSON time series dataset for a nation."""
        code = country_code.upper()
        ext = ".json" if format_type.lower() == "json" else ".csv"
        file_path = os.path.join(self.data_dir, f"{code}{ext}")
        
        if not os.path.exists(file_path):
            alt_ext = ".csv" if ext == ".json" else ".json"
            file_path = os.path.join(self.data_dir, f"{code}{alt_ext}")
            format_type = "csv" if alt_ext == ".csv" else "json"
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset for nation {code} not found in {self.data_dir}")
            
        if format_type.lower() == "json":
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            records = []
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row["latitude"] = float(row["latitude"])
                    row["longitude"] = float(row["longitude"])
                    row["cloud_cover_pct"] = float(row["cloud_cover_pct"])
                    row["satellite_reflectance"] = float(row["satellite_reflectance"])
                    row["optical_depth"] = float(row["optical_depth"])
                    row["humidity_pct"] = float(row["humidity_pct"])
                    row["temperature_c"] = float(row["temperature_c"])
                    row["pressure_hpa"] = float(row["pressure_hpa"])
                    row["wind_speed_ms"] = float(row["wind_speed_ms"])
                    row["cloud_base_altitude_m"] = float(row["cloud_base_altitude_m"])
                    records.append(row)
            return records

    def detect_cloud_formation(self, record: Dict) -> Dict:
        """Detect cloud formation type and spectral properties."""
        reflectance = float(record["satellite_reflectance"])
        optical_depth = float(record["optical_depth"])
        humidity = float(record["humidity_pct"])
        cloud_cover = float(record["cloud_cover_pct"])
        temp = float(record["temperature_c"])
        
        hsv_white_score = min(1.0, max(0.0, (reflectance - 0.05) / 0.85 * 0.7 + (cloud_cover / 100.0) * 0.3))
        
        if cloud_cover < 15.0:
            form = "Clear"
            confidence = 0.98
        elif optical_depth > 15.0 and humidity > 75.0:
            form = "Deep Convection" if temp > 15.0 else "Nimbostratus"
            confidence = 0.93
        elif optical_depth > 6.0 and cloud_cover > 50.0:
            form = "Stratocumulus" if humidity > 65.0 else "Cumulus"
            confidence = 0.90
        elif optical_depth < 3.5 and record["cloud_base_altitude_m"] > 3500:
            form = "Cirrus"
            confidence = 0.88
        else:
            form = "Altocumulus"
            confidence = 0.86
            
        return {
            "timestamp": record["timestamp"],
            "country_code": record["country_code"],
            "country_name": record["country_name"],
            "cloud_cover_pct": cloud_cover,
            "detected_cloud_form": form,
            "hsv_white_score": round(hsv_white_score, 4),
            "confidence": confidence,
            "optical_depth": optical_depth,
            "satellite_reflectance": reflectance,
            "humidity_pct": humidity,
            "pressure_hpa": record["pressure_hpa"]
        }

    def predict_time_series(self, country_code: str, hours: int = 24) -> Dict:
        """Execute sequence forecast for country_code over N hours."""
        records = self.load_nation_records(country_code, format_type="json")
        window = records[-72:] if len(records) >= 72 else records
        
        cloud_seq = [r["cloud_cover_pct"] for r in window]
        refl_seq = [r["satellite_reflectance"] for r in window]
        hum_seq = [r["humidity_pct"] for r in window]
        temp_seq = [r["temperature_c"] for r in window]
        press_seq = [r["pressure_hpa"] for r in window]
        
        cloud_trend = np.polyfit(np.arange(len(cloud_seq)), cloud_seq, 1)[0]
        refl_trend = np.polyfit(np.arange(len(refl_seq)), refl_seq, 1)[0]
        
        last = window[-1]
        last_dt = datetime.fromisoformat(last["timestamp"].replace("Z", "+00:00"))
        
        preds = []
        c_cloud, c_refl, c_hum, c_temp, c_press = cloud_seq[-1], refl_seq[-1], hum_seq[-1], temp_seq[-1], press_seq[-1]
        
        for h in range(1, hours + 1):
            pred_dt = last_dt + timedelta(hours=h)
            diurnal = math.sin((last_dt.hour + h) / 24.0 * 2 * math.pi) * 3.5
            
            n_cloud = min(100.0, max(0.0, c_cloud + cloud_trend * 0.25 + diurnal + np.random.normal(0, 1.2)))
            n_refl = min(1.0, max(0.02, c_refl + refl_trend * 0.15 + (n_cloud - c_cloud) * 0.004))
            n_hum = min(100.0, max(10.0, c_hum - diurnal * 0.4))
            n_temp = c_temp + diurnal * 0.3
            n_press = c_press - (n_cloud - c_cloud) * 0.04
            
            rec = {
                "timestamp": pred_dt.isoformat(),
                "country_code": last["country_code"],
                "country_name": last["country_name"],
                "cloud_cover_pct": n_cloud,
                "satellite_reflectance": n_refl,
                "optical_depth": max(0.1, n_refl * 30.0),
                "humidity_pct": n_hum,
                "temperature_c": n_temp,
                "pressure_hpa": n_press,
                "cloud_base_altitude_m": last["cloud_base_altitude_m"]
            }
            
            det = self.detect_cloud_formation(rec)
            
            preds.append({
                "forecast_hour": h,
                "timestamp": pred_dt.isoformat(),
                "predicted_cloud_cover_pct": round(n_cloud, 2),
                "predicted_cloud_form": det["detected_cloud_form"],
                "predicted_reflectance": round(n_refl, 4),
                "predicted_humidity_pct": round(n_hum, 2),
                "predicted_temperature_c": round(n_temp, 2),
                "predicted_pressure_hpa": round(n_press, 2),
                "confidence_score": round(max(0.72, 0.95 - h * 0.007), 3)
            })
            c_cloud, c_refl, c_hum, c_temp, c_press = n_cloud, n_refl, n_hum, n_temp, n_press
            
        return {
            "country_code": last["country_code"],
            "country_name": last["country_name"],
            "climate_zone": last.get("climate_zone", "Unknown"),
            "forecast_horizon_hours": hours,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "predictions": preds
        }

    async def generate_gemini_reasoning(self, country_name: str, detection: Dict, forecast: Dict) -> str:
        """Generate meteorological reasoning using Gemini or automated expert system."""
        prompt = (
            f"Analyze satellite time-series meteorological prediction for {country_name}.\n"
            f"Current Detection: Cloud Cover = {detection['cloud_cover_pct']}%, Cloud Form = {detection['detected_cloud_form']}, "
            f"Reflectance = {detection['satellite_reflectance']}, Optical Depth = {detection['optical_depth']}.\n"
            f"24h Trend: Cloud cover moving towards {forecast['predictions'][-1]['predicted_cloud_cover_pct']}% with primary form {forecast['predictions'][-1]['predicted_cloud_form']}.\n"
            f"Provide 3 key insights on cloud formation mechanics, aerospace flight hazards, and atmospheric pressure stability."
        )
        
        if self.gemini_model:
            try:
                res = await self.gemini_model.generate_content_async(prompt)
                return res.text
            except Exception:
                pass
                
        # Structured Expert System reasoning output
        cloud_form = detection['detected_cloud_form']
        cover = detection['cloud_cover_pct']
        hazard_level = "High" if cloud_form in ["Deep Convection", "Nimbostratus"] else ("Moderate" if cloud_form in ["Stratocumulus", "Cumulus"] else "Low")
        
        return (
            f"**Meteorological Diagnostic for {country_name}:**\n"
            f"1. **Cloud Dynamics & Formation:** Current observation reveals a dominant {cloud_form} pattern with {cover}% spatial cloud density. "
            f"Reflectance value of {detection['satellite_reflectance']} indicates strong atmospheric moisture condensation.\n"
            f"2. **Aerospace Hazard Index:** [{hazard_level} Risk] {cloud_form} structures are forecasted to evolve over the next 24 hours toward {forecast['predictions'][-1]['predicted_cloud_form']}. "
            f"Pilots should monitor cloud base clearance at {detection.get('cloud_base_altitude_m', 2000)} meters.\n"
            f"3. **Pressure Trend & Stability:** Barometric pressure is holding at {detection['pressure_hpa']} hPa with relative humidity at {detection['humidity_pct']}%. Expect stable diurnal oscillation."
        )
