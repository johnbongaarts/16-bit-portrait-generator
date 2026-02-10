from __future__ import annotations

import json
from pathlib import Path

from app.palettes.snes_snap import snes_snap_palette

PALETTE_DIR = Path(__file__).parent / "data"

_palette_cache: dict | None = None


def _load_all() -> dict:
    """Load all palette JSON files from the data directory."""
    global _palette_cache
    if _palette_cache is not None:
        return _palette_cache

    palettes = {}
    for path in sorted(PALETTE_DIR.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        slug = data["slug"]
        palettes[slug] = data
    _palette_cache = palettes
    return palettes


def get_palette(slug: str) -> dict | None:
    """Get a single palette by slug. Returns None if not found."""
    return _load_all().get(slug)


def get_palette_colors(slug: str, snes_snap: bool = True) -> list[list[int]] | None:
    """Get palette colors, optionally SNES-snapped."""
    palette = get_palette(slug)
    if palette is None:
        return None
    colors = palette["colors"]
    if snes_snap:
        colors = snes_snap_palette(colors)
    return colors


def list_palettes() -> list[dict]:
    """Return all palettes in API response format."""
    palettes = _load_all()
    result = []
    for data in palettes.values():
        result.append({
            "slug": data["slug"],
            "name": data["name"],
            "colors": len(data["colors"]),
            "hex": [
                f"#{r:02x}{g:02x}{b:02x}" for r, g, b in data["colors"]
            ],
            "tags": data.get("tags", []),
        })
    return result
