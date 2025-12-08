"""
Credit Scoring API

FastAPI application for serving credit scoring predictions.

Run with:
    poetry run uvicorn api.app:app --reload --port 8000

Then visit:
    - API docs: http://localhost:8000/docs
    - Health check: http://localhost:8000/health
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import mlflow
import mlflow.sklearn
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

from src.config import MLFLOW_TRACKING_URI, REGISTERED_MODELS
from src.validation import validate_prediction_probabilities, DataValidationError

import logging
import json
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Configure JSON Logging
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
        }
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_record)

logger = logging.getLogger("CreditScoringAPI")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Scoring API",
    description="Machine Learning API for credit default prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware for Request Logging
class RequestLoggerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        log_data = {
            "event": "request_processed",
            "method": request.method,
            "url": str(request.url),
            "status_code": response.status_code,
            "process_time_seconds": round(process_time, 4)
        }
        logger.info(json.dumps(log_data))
        return response

app.add_middleware(RequestLoggerMiddleware)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
model_metadata = {}
EXPECTED_FEATURES = 194  # Updated to match trained model (was 189)


# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================

class PredictionInput(BaseModel):
    """Input schema for single prediction."""
    features: List[float] = Field(
        ...,
        description=f"List of {EXPECTED_FEATURES} feature values in correct order",
        min_items=EXPECTED_FEATURES,
        max_items=EXPECTED_FEATURES
    )
    feature_names: Optional[List[str]] = Field(
        None,
        description="Optional list of feature names for validation"
    )
    client_id: Optional[str] = Field(
        None,
        description="Optional client ID for tracking"
    )

    @validator('features')
    def validate_features_not_nan(cls, v):
        """Validate features don't contain NaN or Inf."""
        arr = np.array(v)
        if np.isnan(arr).any():
            raise ValueError("Features contain NaN values")
        if np.isinf(arr).any():
            raise ValueError("Features contain infinite values")
        return v


class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions."""
    features: List[List[float]] = Field(
        ...,
        description="List of feature arrays for batch prediction"
    )
    client_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of client IDs"
    )

    @validator('features')
    def validate_batch_shape(cls, v):
        """Validate all feature vectors have same length."""
        if not v:
            raise ValueError("Features list is empty")

        lengths = [len(features) for features in v]
        if len(set(lengths)) > 1:
            raise ValueError(f"Inconsistent feature vector lengths: {set(lengths)}")

        if lengths[0] != EXPECTED_FEATURES:
            raise ValueError(
                f"Expected {EXPECTED_FEATURES} features, got {lengths[0]}"
            )

        return v


class PredictionOutput(BaseModel):
    """Output schema for single prediction."""
    prediction: int = Field(..., description="Predicted class (0=no default, 1=default)")
    probability: float = Field(..., description="Probability of default [0-1]")
    risk_level: str = Field(..., description="Risk category (LOW/MEDIUM/HIGH/CRITICAL)")
    client_id: Optional[str] = Field(None, description="Client ID if provided")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions."""
    predictions: List[PredictionOutput]
    count: int = Field(..., description="Number of predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_name: str
    model_version: Optional[str]
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response schema."""
    detail: str
    timestamp: str


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def load_model():
    """Load ML model on startup."""
    global model, model_metadata

    logger.info("Loading credit scoring model...")

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

        # Load production model from registry
        model_name = REGISTERED_MODELS['production']

        try:
            # Try to load from Production stage
            model_uri = f"models:/{model_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            model_metadata['stage'] = 'Production'
            logger.info(f"Loaded model from Production stage: {model_name}")

        except mlflow.exceptions.MlflowException:
            # Fall back to Staging
            logger.warning("Production model not found, trying Staging...")
            model_uri = f"models:/{model_name}/Staging"
            model = mlflow.sklearn.load_model(model_uri)
            model_metadata['stage'] = 'Staging'
            logger.info(f"Loaded model from Staging stage: {model_name}")

        model_metadata.update({
            'name': model_name,
            'uri': model_uri,
            'loaded_at': datetime.now().isoformat()
        })

        logger.info(f"Model loaded successfully: {model_metadata}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        logger.warning("API will start but predictions will fail until model is loaded.")
        model = None
        model_metadata = {'error': str(e)}


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down Credit Scoring API...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Credit Scoring API",
        "version": "1.0.0",
        "status": "active",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_name=model_metadata.get('name', 'unknown'),
        model_version=model_metadata.get('stage', None),
        timestamp=datetime.now().isoformat()
    )


import os
import pandas as pd

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "production_logs.csv"

def log_prediction_to_csv(features: List[float], prediction: int, probability: float, client_id: Optional[str] = None):
    """Log prediction data to CSV for drift monitoring."""
    try:
        data = {
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id,
            "prediction": prediction,
            "probability": probability,
        }
        # Add features with generic names if names aren't available, 
        # but ideally we should map them. For now, we assume simple f_0, f_1...
        # or just store them as a list if we process them later.
        # Better: flattened columns
        for i, val in enumerate(features):
            data[f"feature_{i}"] = val
            
        df = pd.DataFrame([data])
        
        # Append to CSV
        header = not LOG_FILE.exists()
        df.to_csv(LOG_FILE, mode='a', header=header, index=False)
        
    except Exception as e:
        logger.error(f"Failed to log prediction to CSV: {e}")

@app.post(
    "/predict",
    response_model=PredictionOutput,
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Prediction failed"}
    }
)
async def predict(input_data: PredictionInput):
    """
    Single credit scoring prediction.

    Predicts probability of credit default for a single application.

    Args:
        input_data: Prediction input with features

    Returns:
        Prediction output with probability and risk level

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check server logs."
        )

    try:
        # Prepare features
        features = np.array(input_data.features).reshape(1, -1)

        # Make predictions
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0, 1]

        # Validate probability
        try:
            validate_prediction_probabilities(np.array([probability]))
        except DataValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model output validation failed: {e}"
            )

        # Classify risk level
        if probability < 0.2:
            risk_level = "LOW"
        elif probability < 0.4:
            risk_level = "MEDIUM"
        elif probability < 0.6:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
            
        # Log to CSV for monitoring
        log_prediction_to_csv(
            input_data.features, 
            int(prediction), 
            float(probability), 
            input_data.client_id
        )

        return PredictionOutput(
            prediction=int(prediction),
            probability=float(probability),
            risk_level=risk_level,
            client_id=input_data.client_id,
            timestamp=datetime.now().isoformat(),
            model_version=model_metadata.get('stage', 'unknown')
        )

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
    tags=["Prediction"],
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Batch prediction failed"}
    }
)
async def predict_batch(input_data: BatchPredictionInput):
    """
    Batch credit scoring predictions.

    Predicts probabilities of credit default for multiple applications.

    Args:
        input_data: Batch prediction input

    Returns:
        Batch prediction output

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Check server logs."
        )

    try:
        # Prepare features
        features = np.array(input_data.features)

        # Make predictions
        predictions = model.predict(features)
        probabilities = model.predict_proba(features)[:, 1]

        # Validate probabilities
        try:
            validate_prediction_probabilities(probabilities)
        except DataValidationError as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Model output validation failed: {e}"
            )

        # Build response
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            # Classify risk level
            if prob < 0.2:
                risk_level = "LOW"
            elif prob < 0.4:
                risk_level = "MEDIUM"
            elif prob < 0.6:
                risk_level = "HIGH"
            else:
                risk_level = "CRITICAL"

            client_id = None
            if input_data.client_ids and i < len(input_data.client_ids):
                client_id = input_data.client_ids[i]

            results.append(PredictionOutput(
                prediction=int(pred),
                probability=float(prob),
                risk_level=risk_level,
                client_id=client_id,
                timestamp=datetime.now().isoformat(),
                model_version=model_metadata.get('stage', 'unknown')
            ))

        return BatchPredictionOutput(
            predictions=results,
            count=len(results)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Get information about the loaded model.

    Returns:
        Model metadata and configuration
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    return {
        "model_metadata": model_metadata,
        "expected_features": EXPECTED_FEATURES,
        "model_type": type(model).__name__,
        "capabilities": {
            "single_prediction": True,
            "batch_prediction": True,
            "probability_scores": True
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
