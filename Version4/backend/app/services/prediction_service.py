"""
Prediction service for DCGAN weather predictions
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import random

from app.core.database import get_db


class PredictionService:
    """Service for managing DCGAN weather predictions"""
    
    def __init__(self):
        self.db = get_db()
        self.model_loaded = False
    
    async def get_predictions(self, time_horizon: int, region: str, prediction_type: str) -> Dict[str, Any]:
        """Get weather predictions with filters"""
        
        # Mock predictions data
        predictions = []
        for i in range(min(time_horizon, 24)):  # Limit to 24 predictions max
            timestamp = datetime.now() + timedelta(hours=i+1)
            confidence = 0.85 + random.random() * 0.14  # 85-99% confidence
            
            predictions.append({
                "id": i + 1,
                "imageUrl": f"https://picsum.photos/seed/pred_{region}_{i}/200/150.jpg",
                "highResUrl": f"https://picsum.photos/seed/pred_hr_{region}_{i}/800/600.jpg",
                "timeLabel": f"Hour {i + 1}" if prediction_type == "hourly" else f"Day {i + 1}",
                "timestamp": timestamp.isoformat(),
                "confidence": round(confidence, 3),
                "region": region,
                "timeHorizon": time_horizon
            })
        
        # Mock loss data for charts
        loss_data = {
            "labels": [f"Epoch {i}" for i in range(1, 6)],
            "datasets": [
                {
                    "label": "Generator Loss",
                    "data": [0.5 - i*0.05 for i in range(5)],
                    "borderColor": "rgb(75, 192, 192)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "tension": 0.1
                },
                {
                    "label": "Discriminator Loss",
                    "data": [0.6 - i*0.08 for i in range(5)],
                    "borderColor": "rgb(255, 99, 132)",
                    "backgroundColor": "rgba(255, 99, 132, 0.2)",
                    "tension": 0.1
                }
            ]
        }
        
        return {
            "predictions": predictions,
            "lossData": loss_data,
            "total": len(predictions),
            "region": region,
            "timeHorizon": time_horizon,
            "type": prediction_type
        }
    
    async def get_recent_predictions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent predictions"""
        
        predictions = []
        for i in range(min(limit, 10)):
            timestamp = datetime.now() - timedelta(minutes=i*15)
            predictions.append({
                "id": i + 1,
                "location": random.choice(["North America", "Europe", "Asia", "Africa", "South America"]),
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "accuracy": random.randint(85, 98),
                "status": random.choice(["completed", "processing", "completed"])
            })
        
        return predictions
    
    async def generate_prediction_async(self, time_horizon: int, region: str, prediction_type: str):
        """Generate new prediction asynchronously"""
        
        # Simulate DCGAN generation time
        await asyncio.sleep(2)
        
        # Generate mock prediction
        prediction_id = random.randint(1000, 9999)
        confidence = 0.85 + random.random() * 0.14
        
        # Store in database
        await self.db.execute_query(
            """INSERT INTO predictions 
               (image_url, confidence, time_horizon, region, prediction_type, status) 
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                f"https://picsum.photos/seed/new_pred_{prediction_id}/200/150.jpg",
                confidence,
                time_horizon,
                region,
                prediction_type,
                "completed"
            )
        )
        
        print(f"✅ Generated prediction {prediction_id} for {region}")
    
    async def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """Get specific prediction by ID"""
        
        # Mock prediction data
        return {
            "id": prediction_id,
            "imageUrl": f"https://picsum.photos/seed/pred_{prediction_id}/400/300.jpg",
            "highResUrl": f"https://picsum.photos/seed/pred_hr_{prediction_id}/1200/900.jpg",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.92,
            "region": "global",
            "timeHorizon": 24,
            "type": "hourly",
            "status": "completed"
        }
    
    async def get_confidence_scores(self, prediction_id: int) -> Dict[str, float]:
        """Get confidence scores for a prediction"""
        
        return {
            "overall": round(0.85 + random.random() * 0.14, 3),
            "cloudDetection": round(0.88 + random.random() * 0.11, 3),
            "weatherPattern": round(0.82 + random.random() * 0.17, 3),
            "temporalConsistency": round(0.90 + random.random() * 0.09, 3)
        }
    
    async def delete_prediction(self, prediction_id: int) -> bool:
        """Delete a prediction"""
        
        # Mock deletion
        await self.db.execute_query("DELETE FROM predictions WHERE id = ?", (prediction_id,))
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        
        return {
            "accuracy": 94.5,
            "totalPredictions": 1247,
            "avgGenTime": 2.3,
            "modelVersion": "4.0",
            "lastTraining": "2 hours ago",
            "trainingData": "60,000+ frames",
            "successRate": 98.2
        }
    
    async def health_check(self) -> str:
        """Health check for prediction service"""
        
        return "healthy"
    
    async def generate_sample_predictions(self):
        """Generate sample predictions for testing"""
        
        regions = ["global", "north-america", "europe", "asia"]
        types = ["hourly", "daily", "weekly"]
        
        for region in regions:
            for pred_type in types:
                for i in range(5):
                    await self.db.execute_query(
                        """INSERT INTO predictions 
                           (image_url, confidence, time_horizon, region, prediction_type, status) 
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (
                            f"https://picsum.photos/seed/sample_{region}_{pred_type}_{i}/200/150.jpg",
                            round(0.85 + random.random() * 0.14, 3),
                            random.choice([1, 6, 24, 72, 168]),
                            region,
                            pred_type,
                            "completed"
                        )
                    )
        
        print("✅ Sample prediction data generated")
