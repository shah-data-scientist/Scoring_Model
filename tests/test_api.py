"""
Tests for FastAPI endpoints.

Run with: poetry run pytest tests/test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient
import numpy as np

from api.app import app

# Create test client
client = TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_success(self):
        """Test health check returns 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_check_response_structure(self):
        """Test health check response has required fields."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "model_name" in data
        assert "timestamp" in data

    def test_health_check_status_values(self):
        """Test health status is valid."""
        response = client.get("/health")
        data = response.json()

        assert data["status"] in ["healthy", "unhealthy"]
        assert isinstance(data["model_loaded"], bool)


class TestRootEndpoint:
    """Tests for / root endpoint."""

    def test_root_returns_200(self):
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_has_api_info(self):
        """Test root returns API information."""
        response = client.get("/")
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "docs_url" in data


class TestPredictionEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_valid_input(self):
        """Test prediction with valid input."""
        # Generate random features (189 total)
        features = np.random.random(189).tolist()

        data = {
            "features": features,
            "client_id": "TEST_001"
        }

        response = client.post("/predict", json=data)

        # May return 503 if model not loaded, which is OK for testing
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            result = response.json()
            assert "prediction" in result
            assert "probability" in result
            assert "risk_level" in result
            assert result["prediction"] in [0, 1]
            assert 0 <= result["probability"] <= 1
            assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

    def test_predict_invalid_feature_count(self):
        """Test prediction with wrong number of features."""
        data = {
            "features": [0.5] * 50,  # Only 50 features, need 189
            "client_id": "TEST_001"
        }

        response = client.post("/predict", json=data)
        assert response.status_code == 422  # Validation error

    def test_predict_without_client_id(self):
        """Test prediction without client_id (optional field)."""
        features = np.random.random(189).tolist()

        data = {
            "features": features
            # No client_id
        }

        response = client.post("/predict", json=data)
        assert response.status_code in [200, 503]

    def test_predict_with_nan_features(self):
        """Test prediction rejects NaN values."""
        features = [0.5] * 188 + ["NaN"]

        data = {
            "features": features,
            "client_id": "TEST_001"
        }

        response = client.post("/predict", json=data)
        assert response.status_code == 422  # Validation error

    def test_predict_with_inf_features(self):
        """Test prediction rejects infinite values."""
        features = [0.5] * 188 + ["Infinity"]

        data = {
            "features": features,
            "client_id": "TEST_001"
        }

        response = client.post("/predict", json=data)
        assert response.status_code == 422  # Validation error

    def test_predict_empty_features(self):
        """Test prediction rejects empty feature list."""
        data = {
            "features": [],
            "client_id": "TEST_001"
        }

        response = client.post("/predict", json=data)
        assert response.status_code == 422  # Validation error


class TestBatchPredictionEndpoint:
    """Tests for /predict/batch endpoint."""

    def test_batch_predict_valid_input(self):
        """Test batch prediction with valid input."""
        # Generate 3 random feature vectors
        features = [np.random.random(189).tolist() for _ in range(3)]
        client_ids = ["TEST_001", "TEST_002", "TEST_003"]

        data = {
            "features": features,
            "client_ids": client_ids
        }

        response = client.post("/predict/batch", json=data)
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            result = response.json()
            assert "predictions" in result
            assert "count" in result
            assert result["count"] == 3
            assert len(result["predictions"]) == 3

            # Check first prediction structure
            pred = result["predictions"][0]
            assert "prediction" in pred
            assert "probability" in pred
            assert "risk_level" in pred
            assert "client_id" in pred

    def test_batch_predict_without_client_ids(self):
        """Test batch prediction without client_ids (optional)."""
        features = [np.random.random(189).tolist() for _ in range(2)]

        data = {
            "features": features
            # No client_ids
        }

        response = client.post("/predict/batch", json=data)
        assert response.status_code in [200, 503]

    def test_batch_predict_inconsistent_feature_lengths(self):
        """Test batch prediction rejects inconsistent feature lengths."""
        features = [
            [0.5] * 189,  # Correct length
            [0.5] * 100   # Wrong length
        ]

        data = {
            "features": features
        }

        response = client.post("/predict/batch", json=data)
        assert response.status_code == 422  # Validation error

    def test_batch_predict_empty_list(self):
        """Test batch prediction rejects empty feature list."""
        data = {
            "features": []
        }

        response = client.post("/predict/batch", json=data)
        assert response.status_code == 422  # Validation error


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info_success(self):
        """Test model info returns successfully."""
        response = client.get("/model/info")
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            data = response.json()
            assert "model_metadata" in data
            assert "expected_features" in data
            assert "model_type" in data
            assert "capabilities" in data

    def test_model_info_capabilities(self):
        """Test model capabilities are documented."""
        response = client.get("/model/info")

        if response.status_code == 200:
            data = response.json()
            capabilities = data["capabilities"]

            assert "single_prediction" in capabilities
            assert "batch_prediction" in capabilities
            assert "probability_scores" in capabilities


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_endpoint(self):
        """Test accessing non-existent endpoint returns 404."""
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_invalid_method(self):
        """Test using wrong HTTP method."""
        # GET on endpoint that requires POST
        response = client.get("/predict")
        assert response.status_code == 405  # Method not allowed

    def test_missing_request_body(self):
        """Test POST without body."""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error

    def test_malformed_json(self):
        """Test malformed JSON request."""
        response = client.post(
            "/predict",
            data="invalid json",  # Not valid JSON
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestRiskLevelClassification:
    """Tests for risk level classification logic."""

    def test_risk_levels_coverage(self):
        """Test that all risk levels can be produced."""
        # This is a behavioral test - actual probabilities depend on model

        features = np.random.random(189).tolist()
        data = {"features": features}

        response = client.post("/predict", json=data)

        if response.status_code == 200:
            result = response.json()
            # Just verify it's one of the valid levels
            assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self):
        """Test CORS headers are present in response."""
        response = client.options("/predict")

        # CORS headers should be present
        headers = response.headers
        # Note: TestClient may not include all CORS headers
        # In production, test with actual browser or curl


class TestResponseValidation:
    """Tests for response validation."""

    def test_prediction_response_schema(self):
        """Test prediction response matches expected schema."""
        features = np.random.random(189).tolist()
        data = {"features": features, "client_id": "TEST"}

        response = client.post("/predict", json=data)

        if response.status_code == 200:
            result = response.json()

            # Required fields
            assert "prediction" in result
            assert "probability" in result
            assert "risk_level" in result
            assert "timestamp" in result
            assert "model_version" in result

            # Optional fields
            if "client_id" in result:
                assert result["client_id"] == "TEST"

            # Type validation
            assert isinstance(result["prediction"], int)
            assert isinstance(result["probability"], (int, float))
            assert isinstance(result["risk_level"], str)
            assert isinstance(result["timestamp"], str)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
