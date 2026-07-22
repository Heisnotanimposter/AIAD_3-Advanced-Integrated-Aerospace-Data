# 🌟 Version 6: Global Multi-Nation Cloud Intelligence System

Welcome to **Version 6** of the Advanced Integrated Aerospace Data (AIAD) Weather Forecasting Platform.

Version 6 represents a landmark release, expanding our cloud form detection and time-series prediction framework from a single regional dataset to an integrated global dataset spanning **28 popular nations worldwide** (in both CSV and JSON formats).

---

## 🚀 Key Innovations in Version 6

1. **Multi-National Dataset Ingestion (28 Popular Nations)**
   - Integrates complete time-series datasets in both **CSV** and **JSON** formats for: South Korea, USA, Japan, China, Germany, UK, France, India, Brazil, Canada, Australia, Italy, Spain, Russia, South Africa, Mexico, Indonesia, Netherlands, Saudi Arabia, Turkey, Switzerland, Sweden, Argentina, Norway, Singapore, Egypt, UAE, and New Zealand.
   - Comprehensive physical parameters: `timestamp`, `latitude`, `longitude`, `cloud_cover_pct`, `cloud_type`, `satellite_reflectance`, `optical_depth`, `humidity_pct`, `temperature_c`, `pressure_hpa`, `wind_speed_ms`, `cloud_base_altitude_m`.

2. **Unified Cloud Form Detection Engine**
   - Applies HSV spectral equivalent thresholding and optical depth classification to isolate cloud masks and identify 8 distinct cloud forms (`Cumulus`, `Stratus`, `Cirrus`, `Deep Convection`, `Altocumulus`, `Stratocumulus`, `Nimbostratus`, `Clear`).

3. **Time-Series Prediction Pipeline**
   - Implements sequence forecasting models (+24h to +72h horizon) predicting future cloud cover percentages, cloud form transitions, satellite reflectance heatmaps, and atmospheric pressure trajectories.

4. **Multimodal AI Reasoning (Gemini 1.5)**
   - Automatically synthesizes satellite time series predictions into natural language meteorological diagnostics, aerospace flight hazard alerts, and atmospheric stability reports.

5. **Glassmorphism Interactive Dashboard**
   - Sleek UI with 28-nation quick selector, live metric cards, sequence chart visualizations, CSV/JSON download links, and full matrix batch processor.

---

## 🛠️ Quick Start

### 1. Execute Script Suite (Local Python)
Generate/download datasets and run cloud form detection and time series prediction:
```bash
python3 Version6/scripts/download_nations_dataset.py
python3 Version6/scripts/cloud_detector_predictor.py
```

### 2. Launch FastAPI Backend
```bash
cd Version6/backend
pip install -r requirements.txt
python3 main.py
```
*API Swagger UI available at `http://localhost:8000/docs`*

### 3. Deploy via Docker Compose (Recommended)
```bash
cd Version6
docker-compose up -d
```
*Dashboard available at `http://localhost:3000`*

---

## 📁 Architecture Overview
```
Version6/
├── README.md
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── app/
│       ├── api.py
│       └── engines/
│           └── weather_engine_v6.py
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── data/
│   └── nations/ (28 x CSV & JSON files + manifest)
└── scripts/
    ├── download_nations_dataset.py
    └── cloud_detector_predictor.py
```
