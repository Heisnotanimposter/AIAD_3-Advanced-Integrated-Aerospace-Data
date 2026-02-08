try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Satellite Data Analysis"
    VERSION: str = "4.0.0"
    DESCRIPTION: str = "Advanced satellite data analysis with DCGAN predictions"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # CORS Configuration
    ALLOWED_HOSTS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./satellite_data.db"
    
    # Redis Configuration (for caching and background tasks)
    REDIS_URL: str = "redis://localhost:6379"
    
    # Satellite Data Configuration
    SATELLITE_DATA_DIR: str = "./data/satellite"
    MODEL_DIR: str = "./models"
    OUTPUT_DIR: str = "./static/outputs"
    
    # NASA/ESA API Keys (should be set in environment)
    NASA_API_KEY: str = os.getenv("NASA_API_KEY", "")
    ESA_API_KEY: str = os.getenv("ESA_API_KEY", "")
    
    # OpenWeatherMap API Key
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    
    # Model Configuration
    MODEL_VERSION: str = "4.0"
    MODEL_NAME: str = "DCGAN_Satellite_v4"
    
    # GAN Configuration
    VECTOR_NOISE_SHAPE: int = 180
    BATCH_SIZE: int = 12
    EPOCHS: int = 30
    LEARNING_RATE: float = 0.0001
    
    # Image Configuration
    IMAGE_SIZE: tuple = (180, 180)
    IMAGE_CHANNELS: int = 3
    
    # Prediction Configuration
    MAX_PREDICTION_HORIZON: int = 168  # hours (7 days)
    DEFAULT_PREDICTION_HORIZON: int = 24  # hours
    
    # Performance Configuration
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT: int = 30
    
    # Cache Configuration
    CACHE_TTL: int = 3600  # seconds
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Create necessary directories
import os
os.makedirs(settings.SATELLITE_DATA_DIR, exist_ok=True)
os.makedirs(settings.MODEL_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
os.makedirs("./logs", exist_ok=True)
