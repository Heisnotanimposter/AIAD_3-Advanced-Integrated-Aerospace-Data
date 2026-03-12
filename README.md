# 🌍 Advanced Integrated Aerospace Data (AIAD): Weather Platform

Welcome to the AIAD Weather Forecasting project. This repository hosts the complete evolutionary journey of our satellite imagery analysis and weather prediction system, spanning from foundational data preprocessing (Version 1) to our state-of-the-art AI reasoning ecosystem (Version 5).

![Version 5 Dashboard Preview](Version5/frontend/public/favicon.ico) *Placeholder for V5 Screenshot*

---

## 🚀 The AIAD Evolution

This project is organized into distinct versions, each representing a significant leap in our technological capabilities.

### [Phase 1: Version 1 & 2 - The Foundation](./Version2/readme.md)
*   **Focus:** Data Preprocessing & Initial Generation
*   **Key Tech:** OpenCV, TensorFlow, basic DCGAN.
*   **Achievement:** Successfully isolated cloud formations (white regions) from raw satellite video (`resized_KR.mp4`) using HSV thresholding and initiated synthetic frame generation.

### [Phase 2: Version 3 - Visual Analytics](./Version3/index.html)
*   **Focus:** Model Optimization & Visualization
*   **Key Tech:** Time-series GAN, HTML/CSS/JS.
*   **Achievement:** Refined the GAN architecture for better time-series continuity (hourly/daily predictions) and introduced a basic web interface to visualize generation loss and accuracy.

### [Phase 3: Version 4 - Enterprise Architecture](./Version4/README.md)
*   **Focus:** Scalability & Integration
*   **Key Tech:** FastAPI, React, Docker, SQLite.
*   **Achievement:** Rebuilt the system into a robust, modern microservices architecture with a dedicated backend API and a scalable frontend framework.

### 🌟 [Phase 4: Version 5 - Intelligence Integrated](./Version5/README.md) *(Current Release)*
*   **Focus:** Multimodal AI Reasoning & Premium UX
*   **Key Tech:** Google Gemini API, Ant Design, Glassmorphism UI.
*   **Achievement:** Shifted from merely *generating* images to *understanding* them. Version 5 uses the Gemini API to analyze predicted satellite frames, providing human-readable meteorological insights alongside a stunning, newly designed dashboard.

---

## 🚦 Getting Started

### 1. The Interactive Landing Page
For a quick overview and interactive environment check, open the Jupyter Notebook at the root of this project:
```bash
jupyter notebook Earth_engine.ipynb
```

### 2. Launching Version 5 (Recommended)
To run the latest, most powerful version of the platform:

**Prerequisites:** Docker and Docker Compose installed.

```bash
cd Version5
docker-compose up -d
```
*The platform will be available at `http://localhost:3000`*

---

## 📚 Documentation
For detailed guides on the architecture, setup, or AI models, please refer to the `README.md` files located within each specific Version directory.

## 🤝 Contributing
As this represents an evolutionary timeline, contributions are currently focused on **Version 5**. Please see the Version 5 documentation for contribution guidelines.

*AIAD: Shaping the future of aerospace data analysis.*