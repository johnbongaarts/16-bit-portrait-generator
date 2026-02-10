from __future__ import annotations

import numpy as np


def snes_snap_color(r: int, g: int, b: int) -> tuple[int, int, int]:
    """Snap a single 24-bit RGB color to SNES 15-bit color space.

    Each channel is rounded to the nearest multiple of 8,
    producing 32 levels per channel (0, 8, 16, ..., 248)
    for a total of 32,768 possible colors.
    """
    return (
        round(r / 8) * 8,
        round(g / 8) * 8,
        round(b / 8) * 8,
    )


def snes_snap_palette(palette: list[list[int]]) -> list[list[int]]:
    """Snap an entire palette to SNES 15-bit color space."""
    return [list(snes_snap_color(r, g, b)) for r, g, b in palette]


def snes_snap_image(image: np.ndarray) -> np.ndarray:
    """Snap all pixels in an image to SNES 15-bit color space.

    Args:
        image: H x W x 3 uint8 RGB numpy array.

    Returns:
        Image with all channels snapped to multiples of 8.
    """
    # Vectorized: round each channel to nearest multiple of 8
    snapped = np.round(image.astype(np.float32) / 8.0) * 8.0
    return np.clip(snapped, 0, 248).astype(np.uint8)
