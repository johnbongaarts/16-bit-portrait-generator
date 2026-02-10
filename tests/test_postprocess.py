import numpy as np
import pytest

from app.pipeline.postprocess import (
    cleanup_orphans,
    ensure_eye_highlights,
    remove_background,
    upscale_nearest,
)


class TestEyeHighlights:
    def test_injects_when_missing(self):
        """Eye highlight should be injected when no bright pixel exists."""
        # Create a dark image
        img = np.full((8, 8, 3), 30, dtype=np.uint8)
        landmarks = {
            "eye_left": (200, 200),   # Will be scaled to output coords
            "eye_right": (300, 200),
        }
        palette = [[30, 30, 30], [240, 240, 232]]

        result = ensure_eye_highlights(img, landmarks, palette, output_size=8)

        # Should have at least one bright pixel now
        max_lum = 0
        for y in range(8):
            for x in range(8):
                lum = (0.2126 * result[y, x, 0] + 0.7152 * result[y, x, 1] + 0.0722 * result[y, x, 2]) / 255
                max_lum = max(max_lum, lum)
        assert max_lum > 0.5

    def test_preserves_existing_bright(self):
        """Should not inject if bright pixel already exists near eye."""
        img = np.full((8, 8, 3), 30, dtype=np.uint8)
        # Place bright pixel where eye would be
        img[3, 3] = [240, 240, 240]
        landmarks = {"eye_left": (192, 192), "eye_right": (320, 192)}
        palette = [[30, 30, 30], [240, 240, 240]]

        result = ensure_eye_highlights(img, landmarks, palette, output_size=8)

        # Count bright pixels - should not increase much
        bright_count = 0
        for y in range(8):
            for x in range(8):
                if result[y, x, 0] > 200:
                    bright_count += 1
        # Original had 1, should have at most 3 (one existing + up to 2 injected eyes)
        assert bright_count <= 3

    def test_empty_palette_returns_unchanged(self):
        img = np.full((8, 8, 3), 100, dtype=np.uint8)
        result = ensure_eye_highlights(img, {}, [], output_size=8)
        np.testing.assert_array_equal(result, img)


class TestOrphanCleanup:
    def test_removes_isolated_pixel(self):
        """A single pixel surrounded by identical neighbors should be replaced."""
        img = np.full((5, 5, 3), 100, dtype=np.uint8)
        img[2, 2] = [200, 50, 50]  # Orphan in the middle

        result = cleanup_orphans(img)
        np.testing.assert_array_equal(result[2, 2], [100, 100, 100])

    def test_preserves_edges(self):
        """Pixels on a color boundary should NOT be removed."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        img[:, :3] = [100, 100, 100]  # Left half
        img[:, 3:] = [200, 200, 200]  # Right half

        result = cleanup_orphans(img)
        # Edge pixels should be preserved
        np.testing.assert_array_equal(result[2, 2], [100, 100, 100])
        np.testing.assert_array_equal(result[2, 3], [200, 200, 200])

    def test_preserves_corner_pixels(self):
        """Border pixels should not be modified (loop skips them)."""
        img = np.full((5, 5, 3), 100, dtype=np.uint8)
        img[0, 0] = [200, 50, 50]

        result = cleanup_orphans(img)
        np.testing.assert_array_equal(result[0, 0], [200, 50, 50])


class TestUpscaleNearest:
    def test_2x_upscale(self):
        img = np.array([
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ], dtype=np.uint8)

        result = upscale_nearest(img, 2)
        assert result.shape == (4, 4, 3)

        # Top-left 2x2 block should be red
        np.testing.assert_array_equal(result[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(result[0, 1], [255, 0, 0])
        np.testing.assert_array_equal(result[1, 0], [255, 0, 0])
        np.testing.assert_array_equal(result[1, 1], [255, 0, 0])

        # Top-right 2x2 block should be green
        np.testing.assert_array_equal(result[0, 2], [0, 255, 0])

    def test_preserves_no_interpolation(self):
        """Nearest-neighbor should produce only colors from the original."""
        img = np.array([[[100, 150, 200]]], dtype=np.uint8)
        result = upscale_nearest(img, 8)
        assert result.shape == (8, 8, 3)
        for y in range(8):
            for x in range(8):
                np.testing.assert_array_equal(result[y, x], [100, 150, 200])

    def test_1x_is_identity(self):
        img = np.random.randint(0, 256, (4, 4, 3), dtype=np.uint8)
        result = upscale_nearest(img, 1)
        np.testing.assert_array_equal(result, img)


class TestRemoveBackground:
    _landmarks = {
        "eye_left": (80, 90),
        "eye_right": (170, 90),
        "nose": (128, 140),
        "mouth": (128, 180),
    }

    def test_returns_same_shape(self):
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        result = remove_background(img, self._landmarks, threshold=0.5)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def _make_face_image(self):
        """Create a test image with a bright center (face) and dark edges (background)."""
        img = np.full((256, 256, 3), 40, dtype=np.uint8)  # dark background
        # Draw a bright ellipse in the center to simulate a face
        for y in range(256):
            for x in range(256):
                dx = (x - 128) / 80
                dy = (y - 135) / 100
                if dx * dx + dy * dy < 1.0:
                    img[y, x] = [200, 170, 150]  # skin-like color
        return img

    def test_threshold_zero_removes_more(self):
        img = self._make_face_image()
        result_tight = remove_background(img, self._landmarks, threshold=0.0)
        result_loose = remove_background(img, self._landmarks, threshold=1.0)
        black_tight = np.sum(np.all(result_tight == 0, axis=2))
        black_loose = np.sum(np.all(result_loose == 0, axis=2))
        assert black_tight > black_loose

    def test_custom_bg_color(self):
        img = self._make_face_image()
        result = remove_background(img, self._landmarks, threshold=0.0, bg_color=(255, 0, 255))
        magenta_pixels = np.sum(np.all(result == [255, 0, 255], axis=2))
        assert magenta_pixels > 0
