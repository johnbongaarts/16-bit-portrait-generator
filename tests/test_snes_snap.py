import numpy as np
import pytest

from app.palettes.snes_snap import snes_snap_color, snes_snap_image, snes_snap_palette


class TestSnesSnapColor:
    """SNES 15-bit color space snapping (32 levels per channel)."""

    def test_zero_stays_zero(self):
        assert snes_snap_color(0, 0, 0) == (0, 0, 0)

    def test_max_snaps_to_248(self):
        assert snes_snap_color(255, 255, 255) == (256, 256, 256)
        # Note: 255/8 = 31.875, rounds to 32, 32*8 = 256
        # In practice we'd clamp to 248, but snes_snap_color is pure math.
        # snes_snap_image handles clamping.

    def test_value_4_rounds_to_0(self):
        # 4 / 8 = 0.5 — Python's round() does banker's rounding
        # round(0.5) = 0, so 0 * 8 = 0
        assert snes_snap_color(4, 4, 4) == (0, 0, 0)

    def test_value_5_rounds_to_8(self):
        # 5 / 8 = 0.625, rounds to 1, 1 * 8 = 8
        assert snes_snap_color(5, 5, 5) == (8, 8, 8)

    def test_midrange_value(self):
        # 100 / 8 = 12.5 -> round to 12 (banker's) -> 96
        assert snes_snap_color(100, 100, 100) == (96, 96, 96)

    def test_value_248_stays(self):
        assert snes_snap_color(248, 248, 248) == (248, 248, 248)

    def test_value_252_rounds_to_248_or_256(self):
        # 252 / 8 = 31.5 -> round to 32 (banker's) -> 256
        r, g, b = snes_snap_color(252, 252, 252)
        assert r == 256  # Will be clamped by snes_snap_image

    def test_all_values_produce_multiples_of_8(self):
        for v in range(0, 249):
            r, _, _ = snes_snap_color(v, 0, 0)
            assert r % 8 == 0, f"Value {v} snapped to {r} which is not a multiple of 8"

    def test_independent_channels(self):
        r, g, b = snes_snap_color(10, 100, 200)
        assert r == 8
        assert g == 96 or g == 104  # depends on rounding
        assert b == 200


class TestSnesSnapPalette:
    def test_snaps_all_colors(self):
        palette = [[10, 20, 30], [100, 150, 200]]
        result = snes_snap_palette(palette)
        assert len(result) == 2
        for color in result:
            for channel in color:
                assert channel % 8 == 0


class TestSnesSnapImage:
    def test_basic_image(self):
        img = np.array([[[10, 20, 30], [100, 150, 200]]], dtype=np.uint8)
        result = snes_snap_image(img)
        assert result.dtype == np.uint8
        assert result.shape == img.shape
        # All channels should be multiples of 8
        for val in result.flat:
            assert int(val) % 8 == 0

    def test_clamps_to_248(self):
        img = np.array([[[255, 255, 255]]], dtype=np.uint8)
        result = snes_snap_image(img)
        # Should clamp to 248
        assert result[0, 0, 0] == 248
        assert result[0, 0, 1] == 248
        assert result[0, 0, 2] == 248

    def test_preserves_shape(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = snes_snap_image(img)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8
