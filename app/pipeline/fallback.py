from __future__ import annotations

import logging

import cv2
import numpy as np

from app.pipeline.downscale import downscale_area
from app.pipeline.face_detect import detect_and_crop
from app.pipeline.postprocess import (
    apply_dither,
    cleanup_orphans,
    ensure_eye_highlights,
    remove_background,
    upscale_nearest,
)
from app.pipeline.quantize import quantize

logger = logging.getLogger(__name__)


def run_algorithmic_pipeline(
    image: np.ndarray,
    output_size: int = 64,
    n_colors: int = 16,
    palette: list[list[int]] | None = None,
    dither: str = "none",
    snes_snap: bool = True,
    scale: int = 8,
    remove_bg: bool = False,
    bg_threshold: float = 0.5,
) -> tuple[np.ndarray, list[list[int]], dict]:
    """Run the algorithmic (non-GPU) fallback pipeline.

    This is equivalent to the browser-tier pipeline but runs server-side in Python.
    Steps: face detect → crop → pre-process → downscale → quantize → post-process → upscale.

    Args:
        image: Input photo as H x W x 3 uint8 RGB.
        output_size: Output pixel art size (e.g., 64).
        n_colors: Target palette color count.
        palette: Optional predefined palette.
        snes_snap: Snap to SNES 15-bit.
        scale: Upscale factor for final output.

    Returns:
        output_image: Final upscaled image (output_size*scale x output_size*scale x 3).
        palette_used: List of [r, g, b] colors.
        landmarks: Face landmark positions.
    """
    # 1. Face detection and crop
    cropped, landmarks = detect_and_crop(image, target_size=256)

    # 1b. Background removal (optional)
    if remove_bg:
        cropped = remove_background(cropped, landmarks, threshold=bg_threshold)

    # 2. Pre-process: slight blur + contrast/saturation boost
    # Gaussian blur sigma=1
    processed = cv2.GaussianBlur(cropped, (0, 0), 1.0)

    # Contrast boost +10%
    processed = cv2.convertScaleAbs(processed, alpha=1.10, beta=0)

    # Saturation boost +15%
    hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
    processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # 3. Downscale to output size
    small = downscale_area(processed, output_size)

    # 4. Color quantization
    quantized, palette_used = quantize(
        small, n_colors=n_colors, palette=palette, snes_snap=snes_snap
    )

    # 5. Post-processing
    # Eye highlights
    quantized = ensure_eye_highlights(quantized, landmarks, palette_used, output_size)

    # Dithering (non-face regions only)
    quantized = apply_dither(quantized, landmarks, mode=dither, output_size=output_size)

    # Orphan pixel cleanup
    quantized = cleanup_orphans(quantized)

    # 6. Upscale
    output = upscale_nearest(quantized, scale)

    return output, palette_used, landmarks
