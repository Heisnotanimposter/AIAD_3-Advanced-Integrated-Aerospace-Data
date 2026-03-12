from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.api import api_router
import uvicorn

app = FastAPI(
    title="AIAD Weather Platform v5.0",
    description="Advanced Satellite Data Analysis Platform with Gemini AI Integration",
    version="5.0.0"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Welcome to AIAD Weather Platform v5.0 API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "5.0.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
