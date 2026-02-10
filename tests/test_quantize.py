import numpy as np
import pytest

from app.pipeline.quantize import quantize, _rgb_to_oklab, _oklab_to_rgb


class TestOkLabConversion:
    """OkLab color space conversion round-trip tests."""

    def test_black_roundtrip(self):
        black = np.array([[[0, 0, 0]]], dtype=np.uint8)
        lab = _rgb_to_oklab(black)
        rgb = _oklab_to_rgb(lab)
        np.testing.assert_array_almost_equal(rgb, black, decimal=0)

    def test_white_roundtrip(self):
        white = np.array([[[255, 255, 255]]], dtype=np.uint8)
        lab = _rgb_to_oklab(white)
        rgb = _oklab_to_rgb(lab)
        np.testing.assert_allclose(rgb, white, atol=2)

    def test_red_roundtrip(self):
        red = np.array([[[255, 0, 0]]], dtype=np.uint8)
        lab = _rgb_to_oklab(red)
        rgb = _oklab_to_rgb(lab)
        np.testing.assert_allclose(rgb, red, atol=2)

    def test_batch_roundtrip(self):
        colors = np.array([[[128, 64, 200], [0, 255, 128], [50, 50, 50]]], dtype=np.uint8)
        lab = _rgb_to_oklab(colors)
        rgb = _oklab_to_rgb(lab)
        np.testing.assert_allclose(rgb, colors, atol=2)

    def test_oklab_luminance_order(self):
        """Darker colors should have lower L in OkLab."""
        colors = np.array([[[0, 0, 0], [128, 128, 128], [255, 255, 255]]], dtype=np.uint8)
        lab = _rgb_to_oklab(colors)
        assert lab[0, 0, 0] < lab[0, 1, 0] < lab[0, 2, 0]


class TestQuantize:
    def test_reduces_color_count(self):
        # Create image with many colors
        img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        result, palette = quantize(img, n_colors=4, snes_snap=False)
        # Count unique colors in output
        unique = set(map(tuple, result.reshape(-1, 3).tolist()))
        assert len(unique) <= 4

    def test_respects_exact_palette(self):
        img = np.random.randint(0, 256, (8, 8, 3), dtype=np.uint8)
        palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
        result, _ = quantize(img, palette=palette, snes_snap=False)
        unique = set(map(tuple, result.reshape(-1, 3).tolist()))
        palette_set = set(map(tuple, palette))
        assert unique.issubset(palette_set)

    def test_snes_snap_produces_valid_colors(self):
        img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        result, palette = quantize(img, n_colors=8, snes_snap=True)
        # All channels should be multiples of 8, clamped to 0-248
        for r, g, b in np.unique(result.reshape(-1, 3), axis=0):
            assert int(r) % 8 == 0 and 0 <= r <= 248
            assert int(g) % 8 == 0 and 0 <= g <= 248
            assert int(b) % 8 == 0 and 0 <= b <= 248

    def test_output_same_shape(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result, _ = quantize(img, n_colors=16, snes_snap=False)
        assert result.shape == img.shape

    def test_palette_matches_output(self):
        img = np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)
        result, palette = quantize(img, n_colors=8, snes_snap=False)
        unique_colors = set(map(tuple, result.reshape(-1, 3).tolist()))
        # Every color in the output should be close to a palette color
        for color in unique_colors:
            min_dist = min(
                sum((a - b) ** 2 for a, b in zip(color, pc))
                for pc in palette
            )
            assert min_dist < 50, f"Color {color} not close to any palette color"
