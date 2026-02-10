from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from app.palettes.snes_snap import snes_snap_image


def _rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-255 uint8) to OkLab color space.

    OkLab provides perceptually uniform color distances,
    making k-means clustering produce better palettes.
    """
    # Normalize to 0..1
    rgb_f = rgb.astype(np.float32) / 255.0

    # Linearize sRGB
    linear = np.where(rgb_f <= 0.04045, rgb_f / 12.92, ((rgb_f + 0.055) / 1.055) ** 2.4)

    r, g, b = linear[..., 0], linear[..., 1], linear[..., 2]

    # sRGB to LMS (approximate)
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    # Cube root
    l_ = np.cbrt(l)
    m_ = np.cbrt(m)
    s_ = np.cbrt(s)

    # LMS to OkLab
    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    b_ch = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_

    return np.stack([L, a, b_ch], axis=-1)


def _oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert OkLab to RGB (0-255 uint8)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # OkLab to LMS (cube root space)
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    # Cube
    l = l_ ** 3
    m = m_ ** 3
    s = s_ ** 3

    # LMS to linear sRGB
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    b_ch = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s

    rgb_linear = np.stack([r, g, b_ch], axis=-1)

    # sRGB gamma
    rgb_gamma = np.where(
        rgb_linear <= 0.0031308,
        12.92 * rgb_linear,
        1.055 * np.power(np.maximum(rgb_linear, 0), 1.0 / 2.4) - 0.055,
    )

    return np.clip(rgb_gamma * 255.0, 0, 255).astype(np.uint8)


def _nearest_palette_color(pixels_lab: np.ndarray, palette_lab: np.ndarray) -> np.ndarray:
    """Map each pixel to the nearest palette color using Euclidean distance in OkLab.

    This approximates CIEDE2000 well since OkLab is perceptually uniform.
    """
    # pixels_lab: (N, 3), palette_lab: (P, 3)
    # Compute distances: (N, P)
    diffs = pixels_lab[:, np.newaxis, :] - palette_lab[np.newaxis, :, :]
    distances = np.sum(diffs ** 2, axis=-1)
    nearest_idx = np.argmin(distances, axis=-1)
    return nearest_idx


def quantize(
    image: np.ndarray,
    n_colors: int = 16,
    palette: list[list[int]] | None = None,
    snes_snap: bool = True,
) -> tuple[np.ndarray, list[list[int]]]:
    """Quantize image colors using k-means in OkLab space.

    Args:
        image: H x W x 3 uint8 RGB.
        n_colors: Target number of colors (used when palette is None).
        palette: Optional predefined palette [[r,g,b], ...].
        snes_snap: Whether to snap final colors to SNES 15-bit space.

    Returns:
        quantized_image: Same shape as input, with quantized colors.
        palette_used: List of [r, g, b] colors used in the output.
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3)

    # Convert to OkLab for perceptually uniform clustering
    pixels_lab = _rgb_to_oklab(pixels.reshape(1, -1, 3)).reshape(-1, 3)

    if palette is not None:
        # Map to provided palette
        palette_arr = np.array(palette, dtype=np.uint8).reshape(1, -1, 3)
        palette_lab = _rgb_to_oklab(palette_arr).reshape(-1, 3)

        nearest_idx = _nearest_palette_color(pixels_lab, palette_lab)
        palette_rgb = np.array(palette, dtype=np.uint8)
        quantized_pixels = palette_rgb[nearest_idx]
        palette_used = palette
    else:
        # k-means clustering in OkLab space
        kmeans = MiniBatchKMeans(
            n_clusters=min(n_colors, len(pixels)),
            batch_size=1024,
            n_init=3,
            max_iter=100,
            random_state=42,
        )
        labels = kmeans.fit_predict(pixels_lab)
        centroids_lab = kmeans.cluster_centers_

        # Convert centroids back to RGB
        centroids_rgb = _oklab_to_rgb(centroids_lab.reshape(1, -1, 3)).reshape(-1, 3)
        quantized_pixels = centroids_rgb[labels]
        palette_used = centroids_rgb.tolist()

    quantized = quantized_pixels.reshape(h, w, 3)

    if snes_snap:
        quantized = snes_snap_image(quantized)
        # Update palette to reflect snapped colors
        palette_used = [
            [int(round(c / 8) * 8) for c in color]
            for color in palette_used
        ]
        # Clamp
        palette_used = [
            [min(248, max(0, c)) for c in color]
            for color in palette_used
        ]

    return quantized, palette_used
