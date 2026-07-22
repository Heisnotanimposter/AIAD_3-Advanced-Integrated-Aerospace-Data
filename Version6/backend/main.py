import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.api import router as v6_router

app = FastAPI(
    title="AIAD Weather Forecasting API - Version 6",
    description="Global Multi-Nation Cloud Detection & Time-Series Prediction Engine integrating 28+ nations",
    version="6.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v6_router, prefix="/api/v6")

@app.get("/")
async def root():
    return {
        "title": "AIAD Global Weather Forecasting Platform (Version 6)",
        "version": "6.0.0",
        "description": "Cloud Form Detection and Time-Series Prediction for 28+ Popular Nations",
        "docs_url": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
