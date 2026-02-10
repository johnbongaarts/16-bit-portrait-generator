from __future__ import annotations

import numpy as np


def remove_background(
    image: np.ndarray,
    landmarks: dict,
    threshold: float = 0.5,
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Remove background using GrabCut with a landmark-derived face ellipse as prior.

    Args:
        image: H x W x 3 uint8 RGB (the cropped face region, 256 or 512 px).
        landmarks: dict with eye_left, eye_right, nose, mouth as (x, y) tuples
                   in the image's coordinate space (0..image_size).
        threshold: 0.0 = aggressive removal (tight mask), 1.0 = conservative (loose mask).
        bg_color: RGB tuple for background fill. Default black.

    Returns:
        Image with background replaced by bg_color.
    """
    import cv2

    h, w = image.shape[:2]
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    eye_l = landmarks.get("eye_left", (w * 0.35, h * 0.4))
    eye_r = landmarks.get("eye_right", (w * 0.65, h * 0.4))
    mouth = landmarks.get("mouth", (w * 0.5, h * 0.7))

    cx = int((eye_l[0] + eye_r[0]) / 2)
    cy = int((eye_l[1] + mouth[1]) / 2)

    eye_dist = abs(eye_r[0] - eye_l[0])

    # Ellipse radii — threshold scales from tight (0.0) to loose (1.0)
    scale_factor = 0.6 + threshold * 0.6  # range [0.6, 1.2]
    rx = int(eye_dist * scale_factor)
    ry = int(eye_dist * scale_factor * 1.3)

    # GrabCut mask: 0=BG, 1=FG, 2=PR_BG, 3=PR_FG
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Definite foreground ellipse
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, int(cv2.GC_FGD), -1)

    # Probable foreground: slightly larger ellipse
    rx_outer = int(rx * 1.4)
    ry_outer = int(ry * 1.4)
    outer_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(outer_mask, (cx, cy), (rx_outer, ry_outer), 0, 0, 360, 1, -1)
    mask[(outer_mask == 1) & (mask != cv2.GC_FGD)] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(img_bgr, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

    result = image.copy()
    result[fg_mask == 0] = bg_color

    return result


def _luminance_oklab(r: int, g: int, b: int) -> float:
    """Quick approximate luminance using OkLab L channel."""
    # Simplified: just use perceived luminance
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def ensure_eye_highlights(
    image: np.ndarray,
    landmarks: dict,
    palette: list[list[int]],
    output_size: int,
) -> np.ndarray:
    """Inject bright pixel in each eye if not already present.

    This is the single most impactful post-processing step.
    A bright pixel in each eye creates the illusion of life.

    Args:
        image: output_size x output_size x 3 uint8 RGB.
        landmarks: dict with eye_left, eye_right positions (in 512 or 256 coords).
        palette: list of [r, g, b] colors available.
        output_size: the actual image size (for coordinate scaling).

    Returns:
        Modified image with eye highlights.
    """
    if not palette:
        return image

    result = image.copy()

    # Find brightest color in palette
    brightest_idx = max(
        range(len(palette)),
        key=lambda i: _luminance_oklab(*palette[i]),
    )
    brightest = palette[brightest_idx]

    for eye_key in ("eye_left", "eye_right"):
        if eye_key not in landmarks:
            continue

        ex, ey = landmarks[eye_key]

        # Scale landmark coords to output size
        # Landmarks are in the target_size coordinate space (512 or 256)
        # We need to map to output_size
        scale = output_size / 512.0  # assumes landmarks from 512x512 crop
        px = int(round(ex * scale))
        py = int(round(ey * scale))

        # Clamp to image bounds
        px = max(0, min(output_size - 1, px))
        py = max(0, min(output_size - 1, py))

        # Check if there's already a bright pixel within 1px radius
        has_bright = False
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = px + dx, py + dy
                if 0 <= nx < output_size and 0 <= ny < output_size:
                    r, g, b = result[ny, nx]
                    if _luminance_oklab(int(r), int(g), int(b)) > 0.7:
                        has_bright = True
                        break
            if has_bright:
                break

        if not has_bright:
            result[py, px] = brightest

    return result


def cleanup_orphans(image: np.ndarray) -> np.ndarray:
    """Remove isolated single pixels that create visual noise.

    An orphan pixel is one whose color differs from ALL 4 cardinal neighbors
    AND all 4 neighbors share the same color. Replace orphan with surrounding color.
    """
    h, w = image.shape[:2]
    result = image.copy()

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            current = result[y, x]

            # Get 4 cardinal neighbors
            top = result[y - 1, x]
            bottom = result[y + 1, x]
            left = result[y, x - 1]
            right = result[y, x + 1]

            # Check if all 4 neighbors are the same color
            if (
                np.array_equal(top, bottom)
                and np.array_equal(top, left)
                and np.array_equal(top, right)
            ):
                # Check if current pixel differs from neighbors
                if not np.array_equal(current, top):
                    result[y, x] = top

    return result


def apply_dither(
    image: np.ndarray,
    landmarks: dict,
    mode: str = "none",
    output_size: int = 64,
) -> np.ndarray:
    """Apply dithering to non-face regions only.

    Args:
        image: output_size x output_size x 3 uint8 RGB (quantized).
        landmarks: face landmark positions.
        mode: "none", "bayer2x2", "bayer4x4", "floyd-steinberg".
        output_size: actual image size.

    Returns:
        Dithered image.
    """
    if mode == "none":
        return image

    # Create face mask (ellipse around face center)
    mask = np.zeros((output_size, output_size), dtype=bool)
    if landmarks:
        # Approximate face region from eye positions
        eye_l = landmarks.get("eye_left", (output_size * 0.35, output_size * 0.4))
        eye_r = landmarks.get("eye_right", (output_size * 0.65, output_size * 0.4))

        scale = output_size / 512.0
        cx = int((eye_l[0] + eye_r[0]) / 2 * scale)
        cy = int((eye_l[1] + eye_r[1]) / 2 * scale)
        rx = int(abs(eye_r[0] - eye_l[0]) * scale * 1.2)
        ry = int(rx * 1.3)

        for y in range(output_size):
            for x in range(output_size):
                if rx > 0 and ry > 0:
                    if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
                        mask[y, x] = True

    # For now, only Bayer dithering is applied as a subtle pattern overlay
    # Floyd-Steinberg would require re-quantization which is complex
    if mode in ("bayer2x2", "bayer4x4"):
        result = image.copy().astype(np.float32)

        if mode == "bayer2x2":
            bayer = np.array([[0, 2], [3, 1]], dtype=np.float32) / 4.0 - 0.5
            size = 2
        else:
            bayer = np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5],
            ], dtype=np.float32) / 16.0 - 0.5
            size = 4

        # Apply bayer pattern to non-face regions
        for y in range(output_size):
            for x in range(output_size):
                if not mask[y, x]:
                    offset = bayer[y % size, x % size] * 16  # subtle
                    result[y, x] = np.clip(result[y, x] + offset, 0, 255)

        return result.astype(np.uint8)

    return image


def upscale_nearest(image: np.ndarray, scale: int) -> np.ndarray:
    """Upscale image using nearest-neighbor interpolation.

    This preserves pixel crispness — no interpolation artifacts.
    """
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
