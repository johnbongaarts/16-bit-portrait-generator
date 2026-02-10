from __future__ import annotations

import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Singleton landmarker instance
_landmarker = None


class NoFaceDetectedError(Exception):
    """Raised when no face is found in the input image."""
    pass


def _get_mediapipe():
    """Lazy import of mediapipe to avoid hard dep at module level."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        FaceLandmarker,
        FaceLandmarkerOptions,
        RunningMode,
    )
    return mp, BaseOptions, FaceLandmarker, FaceLandmarkerOptions, RunningMode


def _get_landmarker():
    """Initialize the MediaPipe FaceLandmarker singleton."""
    global _landmarker
    if _landmarker is not None:
        return _landmarker

    mp, BaseOptions, FaceLandmarker, FaceLandmarkerOptions, RunningMode = _get_mediapipe()

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path="face_landmarker_v2_with_blendshapes.task"
        ),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _landmarker = FaceLandmarker.create_from_options(options)
    return _landmarker


def init_landmarker(model_path: str) -> None:
    """Initialize the landmarker with a specific model path."""
    global _landmarker
    mp, BaseOptions, FaceLandmarker, FaceLandmarkerOptions, RunningMode = _get_mediapipe()

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _landmarker = FaceLandmarker.create_from_options(options)
    logger.info("MediaPipe FaceLandmarker initialized from %s", model_path)


def _detect_and_crop_single(
    image: np.ndarray, target_size: int = 512
) -> tuple[np.ndarray, dict]:
    """Core detection on a single image orientation. No retries.

    Raises NoFaceDetectedError if no face is found.
    """
    import mediapipe as mp

    landmarker = _get_landmarker()

    h, w = image.shape[:2]

    # Run detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        raise NoFaceDetectedError("No face detected in the uploaded image.")

    face = result.face_landmarks[0]

    # Extract key landmark positions (pixel coords)
    def lm_px(idx):
        return (face[idx].x * w, face[idx].y * h)

    # Jaw landmarks for bounding box: 10 (top), 152 (bottom), 234 (left), 454 (right)
    jaw_indices = [10, 152, 234, 454]
    xs = [face[i].x * w for i in jaw_indices]
    ys = [face[i].y * h for i in jaw_indices]

    # Also include all landmarks for better bbox
    for i in range(len(face)):
        xs.append(face[i].x * w)
        ys.append(face[i].y * h)

    bbox_x1 = min(xs)
    bbox_y1 = min(ys)
    bbox_x2 = max(xs)
    bbox_y2 = max(ys)

    # Expand bbox by 35% on all sides
    bbox_w = bbox_x2 - bbox_x1
    bbox_h = bbox_y2 - bbox_y1
    pad_x = bbox_w * 0.35
    pad_y = bbox_h * 0.35

    crop_x1 = max(0, int(bbox_x1 - pad_x))
    crop_y1 = max(0, int(bbox_y1 - pad_y))
    crop_x2 = min(w, int(bbox_x2 + pad_x))
    crop_y2 = min(h, int(bbox_y2 + pad_y))

    # Eye landmarks for alignment: 33 (left eye outer), 263 (right eye outer)
    eye_left = lm_px(33)
    eye_right = lm_px(263)

    # Calculate rotation angle to level eyes
    dy = eye_right[1] - eye_left[1]
    dx = eye_right[0] - eye_left[0]
    angle = math.degrees(math.atan2(dy, dx))

    # Rotate image around center of the face region
    center_x = (crop_x1 + crop_x2) / 2
    center_y = (crop_y1 + crop_y2) / 2
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

    # Crop the rotated image
    cropped = rotated[crop_y1:crop_y2, crop_x1:crop_x2]

    # Resize to target size
    cropped = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Map landmark positions to cropped coordinates
    def map_landmark(idx):
        px, py = lm_px(idx)
        # Apply rotation
        cos_a = math.cos(math.radians(-angle))
        sin_a = math.sin(math.radians(-angle))
        rx = cos_a * (px - center_x) - sin_a * (py - center_y) + center_x
        ry = sin_a * (px - center_x) + cos_a * (py - center_y) + center_y
        # Map to crop coordinates
        cx = (rx - crop_x1) / (crop_x2 - crop_x1) * target_size
        cy = (ry - crop_y1) / (crop_y2 - crop_y1) * target_size
        return (cx, cy)

    landmarks = {
        "eye_left": map_landmark(33),
        "eye_right": map_landmark(263),
        "nose": map_landmark(1),
        "mouth": map_landmark(13),
    }

    return cropped, landmarks


def detect_and_crop(
    image: np.ndarray, target_size: int = 512
) -> tuple[np.ndarray, dict]:
    """Detect a face, crop with padding, align eyes, and resize.

    On failure, mirrors the image horizontally and retries up to 2 times.
    This handles side profiles and angled shots where one eye is visible
    but MediaPipe doesn't detect the face in the original orientation.

    Args:
        image: H x W x 3 uint8 RGB numpy array.
        target_size: Output size (square). 512 for GPU pipeline, 256 for browser-like.

    Returns:
        cropped_face: target_size x target_size x 3 uint8 RGB array.
        landmarks: dict with eye_left, eye_right, nose, mouth positions
                   normalized to cropped image coordinates (0..target_size).

    Raises:
        NoFaceDetectedError: If no face is detected after all attempts.
    """
    MAX_MIRROR_RETRIES = 2
    current = image

    for attempt in range(MAX_MIRROR_RETRIES + 1):
        try:
            cropped, landmarks = _detect_and_crop_single(current, target_size)

            if attempt % 2 == 1:
                # Detected on a mirrored image — flip the crop back to
                # match the original orientation
                cropped = np.fliplr(cropped).copy()
                # Mirror landmark x coords and swap left/right eyes
                landmarks = {
                    "eye_left": (target_size - landmarks["eye_right"][0], landmarks["eye_right"][1]),
                    "eye_right": (target_size - landmarks["eye_left"][0], landmarks["eye_left"][1]),
                    "nose": (target_size - landmarks["nose"][0], landmarks["nose"][1]),
                    "mouth": (target_size - landmarks["mouth"][0], landmarks["mouth"][1]),
                }

            logger.info(
                "Face detected on attempt %d%s",
                attempt + 1,
                " (mirrored)" if attempt % 2 == 1 else "",
            )
            return cropped, landmarks

        except NoFaceDetectedError:
            if attempt == MAX_MIRROR_RETRIES:
                raise
            # Mirror for the next attempt
            current = np.fliplr(current).copy()
            logger.info(
                "No face on attempt %d, mirroring and retrying...", attempt + 1
            )
