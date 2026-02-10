"""Tests for face detection module.

Note: Full face detection tests require the MediaPipe model file.
These tests verify the module structure and error handling.
Integration tests with actual face images should use the test fixture.
"""

import numpy as np
import pytest

from app.pipeline.face_detect import NoFaceDetectedError


class TestNoFaceDetectedError:
    def test_is_exception(self):
        assert issubclass(NoFaceDetectedError, Exception)

    def test_message(self):
        err = NoFaceDetectedError("test message")
        assert str(err) == "test message"


class TestFaceDetectModule:
    """Structural tests that don't require the model."""

    def test_imports(self):
        from app.pipeline.face_detect import detect_and_crop, init_landmarker
        assert callable(detect_and_crop)
        assert callable(init_landmarker)

    def test_detect_requires_mediapipe(self):
        """detect_and_crop should raise if mediapipe not installed or no model loaded."""
        img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(Exception):
            from app.pipeline.face_detect import detect_and_crop
            detect_and_crop(img)
