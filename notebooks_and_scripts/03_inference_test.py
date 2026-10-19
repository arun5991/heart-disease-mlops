import json
import requests

API_URL = "http://localhost:8000"

SAMPLE_PATIENTS = [
    {
        "name": "Patient A (High Risk Profile)",
        "payload": {
            "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
            "fbs": 0, "restecg": 2, "thalach": 108, "exang": 1,
            "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3
        },
    },
    {
        "name": "Patient B (Low Risk Profile)",
        "payload": {
            "age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204,
            "fbs": 0, "restecg": 2, "thalach": 172, "exang": 0,
            "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 3
        },
    },
]


def run_inference_checks():
    print("1. Checking /health endpoint...")
    health_resp = requests.get(f"{API_URL}/health", timeout=5)
    print(f"Status: {health_resp.status_code} | Response: {health_resp.json()}\n")

    print("2. Sending sample patient requests to /predict...")
    for sample in SAMPLE_PATIENTS:
        resp = requests.post(f"{API_URL}/predict", json=sample["payload"], timeout=5)
        print(f"--- {sample['name']} ---")
        print(json.dumps(resp.json(), indent=2))

    print("\n3. Verifying Prometheus /metrics endpoint...")
    metrics_resp = requests.get(f"{API_URL}/metrics", timeout=5)
    for line in metrics_resp.text.splitlines():
        if line.startswith(("api_requests_total", "model_predictions_total")):
            print(" ", line)


if __name__ == "__main__":
    run_inference_checks()