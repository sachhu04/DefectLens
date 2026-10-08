import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data

def test_categories_endpoint(client):
    response = client.get("/categories")
    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "bottle" in data["categories"]
    assert "cable" in data["categories"]
    assert "metal_nut" in data["categories"]

def test_benchmark_endpoint_all_categories(client):
    for cat in ["bottle", "cable", "metal_nut"]:
        response = client.get(f"/benchmark?category={cat}")
        assert response.status_code == 200
        data = response.json()
        assert data["category"] == cat
        assert data["speedup_factor"] >= 1.4
        assert data["token_reduction_pct"] > 70.0
        assert "baseline" in data
        assert "mctf" in data

def test_model_info_endpoint(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "mctf" in data["models"]
    assert "baseline" in data["models"]

def test_benchmark_endpoint_bottle(client):
    response = client.get("/benchmark?category=bottle")
    assert response.status_code == 200
    data = response.json()
    assert data["category"] == "bottle"
    assert data["speedup_factor"] >= 1.5
    assert data["token_reduction_pct"] > 70.0
    assert "baseline" in data
    assert "mctf" in data
    assert "comparisons" in data
    assert len(data["comparisons"]) == 5
    assert "layer_token_distribution" in data
    assert len(data["layer_token_distribution"]) > 0

def test_benchmark_summary_csv_endpoint(client):
    response = client.get("/benchmark/summary-csv")
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert "bottle" in data["content"]
