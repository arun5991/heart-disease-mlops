import logging
import os
import time
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

# Configure structured logging for API requests (Task 8)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/api_requests.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("heart_disease_api")

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/models/heart_disease_pipeline.joblib")

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production MLOps API serving the UCI Heart Disease classifier with Prometheus metrics.",
    version="1.0.0",
)

# Prometheus Metrics (Task 8)
REQUEST_COUNT = Counter(
    "api_requests_total",
    "Total number of API requests",
    ["endpoint", "method", "http_status"],
)
PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total predictions grouped by predicted class",
    ["predicted_label"],
)
REQUEST_LATENCY = Histogram(
    "api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
)


class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120, example=63)
    sex: int = Field(..., ge=0, le=1, example=1)
    cp: int = Field(..., ge=1, le=4, example=3)
    trestbps: float = Field(..., ge=50, le=260, example=145)
    chol: float = Field(..., ge=80, le=650, example=233)
    fbs: int = Field(..., ge=0, le=1, example=1)
    restecg: int = Field(..., ge=0, le=2, example=0)
    thalach: float = Field(..., ge=50, le=250, example=150)
    exang: int = Field(..., ge=0, le=1, example=0)
    oldpeak: float = Field(..., ge=0.0, le=10.0, example=2.3)
    slope: int = Field(..., ge=1, le=3, example=1)
    ca: int = Field(..., ge=0, le=4, example=0)
    thal: int = Field(..., example=6)


class PredictionResponse(BaseModel):
    prediction: int
    risk_category: str
    confidence: float
    disease_probability: float


def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run 02_train_and_track.py first."
        )
    return joblib.load(MODEL_PATH)


@app.middleware("http")
async def log_and_monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    elapsed = time.time() - start_time
    endpoint = request.url.path
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(elapsed)
    REQUEST_COUNT.labels(
        endpoint=endpoint, method=request.method, http_status=response.status_code
    ).inc()
    logger.info(
        f"Method={request.method} Path={endpoint} Status={response.status_code} "
        f"Latency={elapsed:.4f}s"
    )
    return response


@app.get("/health")
def health_check():
    model_exists = os.path.exists(MODEL_PATH)
    return {"status": "healthy" if model_exists else "degraded", "model_loaded": model_exists}


@app.post("/predict", response_model=PredictionResponse)
def predict_heart_disease(patient: PatientData):
    try:
        model = get_model()
        input_df = pd.DataFrame([patient.model_dump()])
        prob_disease = float(model.predict_proba(input_df)[0, 1])
        pred_class = int(prob_disease >= 0.5)
        confidence = prob_disease if pred_class == 1 else (1.0 - prob_disease)
        risk_label = "High Risk (Heart Disease Present)" if pred_class == 1 else "Low Risk (Normal)"

        PREDICTION_COUNT.labels(predicted_label=str(pred_class)).inc()
        logger.info(
            f"Prediction request | Input={patient.model_dump()} | "
            f"Prediction={pred_class} | Confidence={confidence:.4f}"
        )

        return PredictionResponse(
            prediction=pred_class,
            risk_category=risk_label,
            confidence=round(confidence, 4),
            disease_probability=round(prob_disease, 4),
        )
    except Exception as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
