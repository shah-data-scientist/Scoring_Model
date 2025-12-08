# Project 8: Deploy and Monitor Scoring Model - Implementation Plan

## 📋 Context & Objectives
**Goal:** Take the "Credit Express" scoring model developed in the previous phase and deploy it into a production-ready environment.
**Focus:** Robust API, Containerization (Docker), CI/CD Automation, and Model Monitoring (Data Drift).

---

## 🏗️ Architecture & Tech Stack

- **Model:** LightGBM (from Project 6/7)
- **API Framework:** FastAPI (Python)
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Monitoring:** Evidently AI (for Data Drift) + Streamlit (Dashboard)
- **Storage:** Local file system (simulating production logs) or SQLite

---

## 📅 Detailed Implementation Plan

### Phase 1: API Robustness & Containerization
*Goal: Ensure the API is production-ready and portable.*

1.  **Review API (`api/app.py`):**
    - [ ] Ensure model is loaded **only once** at startup (lifespan events).
    - [ ] Validate inputs strictness (Pydantic).
    - [ ] Handle edge cases: `NaN`, missing fields, wrong types.
    - [ ] Add structured logging (JSON format preferred) for all requests.
2.  **Docker Setup:**
    - [ ] Create `Dockerfile`:
        - Base image: `python:3.11-slim`
        - Install dependencies (poetry or requirements.txt).
        - Copy code and model artifacts.
        - Expose port 8000.
    - [ ] Create `.dockerignore` to exclude `mlruns`, `logs`, `venv`.
    - [ ] Test building and running the container locally.

### Phase 2: CI/CD Pipeline
*Goal: Automate testing and deployment.*

1.  **GitHub Actions Workflow (`.github/workflows/main.yml`):**
    - [ ] **Trigger:** On push to `main` or pull request.
    - [ ] **Job 1: Test:**
        - Check out code.
        - Install dependencies.
        - Run `pytest` (Unit tests + API integration tests).
    - [ ] **Job 2: Build & Push (Optional/Simulated):**
        - Build Docker image.
        - (Mock) Push to registry (Docker Hub) or deploy to a PaaS (Render/Heroku) if credentials available.

### Phase 3: Monitoring & Data Drift
*Goal: Detect when production data deviates from training data.*

1.  **Logging Strategy:**
    - [ ] Update API to log every prediction request (features + result) to a "production_logs.csv" or SQLite DB.
2.  **Drift Detection System:**
    - [ ] Create a monitoring script (`scripts/monitor_drift.py`) using **Evidently AI**.
    - [ ] Compare `reference` data (Training set) vs `current` data (Production logs).
    - [ ] Generate a Drift Report (HTML/JSON).
3.  **Monitoring Dashboard:**
    - [ ] Create a simple Streamlit view to display the latest Drift Report.

### Phase 4: Performance Optimization
*Goal: Measure and improve efficiency.*

1.  **Profiling:**
    - [ ] Measure API latency (response time) and memory usage.
    - [ ] Use `cProfile` or simple timing decorators.
2.  **Optimization:**
    - [ ] Test ONNX runtime (optional) or quantization if simple optimization isn't enough.
    - [ ] Document the "Before vs After" metrics.

### Phase 5: Documentation & Delivery
1.  **README:** Detailed instructions on:
    - How to build/run Docker.
    - How to trigger CI/CD.
    - How to view Monitoring.
2.  **Final Checks:**
    - Verify all tests pass.
    - Verify Docker container works.
    - Verify Drift report generation.

---

## 🛠️ Immediate Next Steps
1.  **Refactor API:** Implement robust logging and single-load pattern.
2.  **Dockerize:** Build the image.
3.  **Automate:** Write the GitHub Actions YAML.