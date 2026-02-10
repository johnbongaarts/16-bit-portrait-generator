import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_returns_200(self):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["pipeline"] == "algorithmic"
        assert data["version"] == "0.2.0"


class TestPalettesEndpoint:
    def test_returns_palettes(self):
        response = client.get("/api/palettes")
        assert response.status_code == 200
        data = response.json()
        assert "palettes" in data
        assert len(data["palettes"]) >= 5

    def test_palette_structure(self):
        response = client.get("/api/palettes")
        data = response.json()
        for p in data["palettes"]:
            assert "slug" in p
            assert "name" in p
            assert "colors" in p
            assert "hex" in p
            assert "tags" in p


class TestGenerateEndpoint:
    def _make_test_image(self, width=256, height=256) -> bytes:
        """Create a simple test image."""
        img = Image.fromarray(
            np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        )
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.getvalue()

    def test_invalid_output_size(self):
        img_bytes = self._make_test_image()
        response = client.post(
            "/api/generate",
            files={"image": ("test.png", img_bytes, "image/png")},
            data={"output_size": "100"},  # Invalid
        )
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_parameter"

    def test_invalid_palette_colors(self):
        img_bytes = self._make_test_image()
        response = client.post(
            "/api/generate",
            files={"image": ("test.png", img_bytes, "image/png")},
            data={"palette_colors": "10"},  # Invalid
        )
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_parameter"

    def test_invalid_dither_mode(self):
        img_bytes = self._make_test_image()
        response = client.post(
            "/api/generate",
            files={"image": ("test.png", img_bytes, "image/png")},
            data={"dither": "invalid"},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_parameter"

    def test_oversized_image(self):
        # Create a >10MB payload
        large_data = b"\x00" * (11 * 1024 * 1024)
        response = client.post(
            "/api/generate",
            files={"image": ("test.png", large_data, "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["detail"]["error"] == "image_too_large"

    def test_unknown_palette_name(self):
        img_bytes = self._make_test_image()
        response = client.post(
            "/api/generate",
            files={"image": ("test.png", img_bytes, "image/png")},
            data={"palette_name": "nonexistent-palette"},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_parameter"

    def test_invalid_bg_threshold_rejected(self):
        img_bytes = self._make_test_image()
        response = client.post(
            "/api/generate",
            files={"image": ("test.png", img_bytes, "image/png")},
            data={"bg_threshold": "1.5"},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["error"] == "invalid_parameter"

    def test_remove_bg_accepts_boolean(self):
        img_bytes = self._make_test_image()
        response = client.post(
            "/api/generate",
            files={"image": ("test.png", img_bytes, "image/png")},
            data={"remove_bg": "true", "bg_threshold": "0.5"},
        )
        # Should not fail on parameter validation (may fail later in pipeline)
        assert response.status_code != 422
