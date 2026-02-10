import json
from pathlib import Path

import pytest

from app.palettes.loader import get_palette, get_palette_colors, list_palettes
from app.palettes.snes_snap import snes_snap_palette

PALETTE_DIR = Path(__file__).parent.parent / "app" / "palettes" / "data"


class TestPaletteLoading:
    def test_all_json_files_load(self):
        """Every palette JSON file should load without error."""
        for path in PALETTE_DIR.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            assert "slug" in data
            assert "name" in data
            assert "colors" in data
            assert isinstance(data["colors"], list)
            assert len(data["colors"]) > 0

    def test_color_counts_match(self):
        """Each palette's color array length should be reasonable."""
        for path in PALETTE_DIR.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            colors = data["colors"]
            # Each color should be [r, g, b]
            for color in colors:
                assert len(color) == 3, f"Palette {data['slug']}: color {color} is not [r,g,b]"
                for channel in color:
                    assert 0 <= channel <= 255, f"Palette {data['slug']}: channel {channel} out of range"

    def test_get_palette_by_slug(self):
        p = get_palette("jehkoba32")
        assert p is not None
        assert p["slug"] == "jehkoba32"
        assert len(p["colors"]) == 32

    def test_get_unknown_palette_returns_none(self):
        assert get_palette("nonexistent-palette") is None

    def test_get_palette_colors_snes_snapped(self):
        colors = get_palette_colors("jehkoba32", snes_snap=True)
        assert colors is not None
        for r, g, b in colors:
            assert r % 8 == 0 and 0 <= r <= 248
            assert g % 8 == 0 and 0 <= g <= 248
            assert b % 8 == 0 and 0 <= b <= 248

    def test_list_palettes_returns_all(self):
        palettes = list_palettes()
        slugs = {p["slug"] for p in palettes}
        assert "jehkoba32" in slugs
        assert "chrono-trigger" in slugs
        assert "ff6-portraits" in slugs
        assert "snes-warm" in slugs
        assert "snes-cool" in slugs

    def test_list_palettes_has_hex(self):
        palettes = list_palettes()
        for p in palettes:
            assert "hex" in p
            assert len(p["hex"]) == p["colors"]
            for hex_color in p["hex"]:
                assert hex_color.startswith("#")
                assert len(hex_color) == 7


class TestBuiltInPalettes:
    """Verify specific palette properties."""

    def test_chrono_trigger_has_48_colors(self):
        p = get_palette("chrono-trigger")
        assert len(p["colors"]) == 48

    def test_ff6_has_32_colors(self):
        p = get_palette("ff6-portraits")
        assert len(p["colors"]) == 32

    def test_jehkoba_has_32_colors(self):
        p = get_palette("jehkoba32")
        assert len(p["colors"]) == 32

    def test_snes_warm_has_24_colors(self):
        p = get_palette("snes-warm")
        assert len(p["colors"]) == 24

    def test_snes_cool_has_24_colors(self):
        p = get_palette("snes-cool")
        assert len(p["colors"]) == 24
