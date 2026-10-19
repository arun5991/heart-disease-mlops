import os
import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.app import app
from src.data_processor import (
    ClinicalFeatureEngineer,
    FEATURE_COLUMNS,
    build_preprocessor,
    load_and_clean_data,
)


@pytest.fixture(scope="module")
def cleaned_df():
    return load_and_clean_data()


def test_data_cleaning_and_schema(cleaned_df):
    """Verify dataset has no nulls, expected columns, and binary target."""
    assert not cleaned_df.isnull().any().any(), "Cleaned dataset must not contain NaNs"
    for col in FEATURE_COLUMNS + ["target"]:
        assert col in cleaned_df.columns
    assert set(cleaned_df["target"].unique()).issubset({0, 1})
    assert len(cleaned_df) >= 250


def test_feature_engineer_transformer(cleaned_df):
    """Verify ClinicalFeatureEngineer adds the two domain ratio features."""
    engineer = ClinicalFeatureEngineer()
    transformed = engineer.fit_transform(cleaned_df[FEATURE_COLUMNS].iloc[:5])
    assert "rate_pressure_proxy" in transformed.columns
    assert "chol_age_ratio" in transformed.columns


def test_model_pipeline_fit_and_predict(cleaned_df, tmp_path):
    """Verify preprocessing + classifier pipeline trains, predicts probabilities, and serializes."""
    X = cleaned_df[FEATURE_COLUMNS]
    y = cleaned_df["target"]

    pipe = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe.fit(X, y)
    preds = pipe.predict(X.iloc[:10])
    probs = pipe.predict_proba(X.iloc[:10])

    assert len(preds) == 10
    assert probs.shape == (10, 2)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()

    # Ensure model exists for API test
    os.makedirs("artifacts/models", exist_ok=True)
    if not os.path.exists("artifacts/models/heart_disease_pipeline.joblib"):
        joblib.dump(pipe, "artifacts/models/heart_disease_pipeline.joblib")


def test_fastapi_predict_and_health(cleaned_df):
    """Verify FastAPI /health, /predict, and /metrics endpoints."""
    client = TestClient(app)

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "healthy"

    sample_payload = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 1, "ca": 0, "thal": 6
    }
    resp = client.post("/predict", json=sample_payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["confidence"] <= 1.0

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "api_requests_total" in metrics_resp.text
