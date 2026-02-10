from __future__ import annotations

import cv2
import numpy as np


def downscale_nearest(image: np.ndarray, target_size: int) -> np.ndarray:
    """Downscale image using nearest-neighbor interpolation.

    Args:
        image: H x W x 3 uint8 RGB.
        target_size: Output width/height in pixels.

    Returns:
        target_size x target_size x 3 uint8 RGB.
    """
    return cv2.resize(
        image,
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST,
    )


def downscale_area(image: np.ndarray, target_size: int) -> np.ndarray:
    """Downscale using area averaging (good for initial downscale before quantization).

    Args:
        image: H x W x 3 uint8 RGB.
        target_size: Output width/height in pixels.

    Returns:
        target_size x target_size x 3 uint8 RGB.
    """
    return cv2.resize(
        image,
        (target_size, target_size),
        interpolation=cv2.INTER_AREA,
    )
