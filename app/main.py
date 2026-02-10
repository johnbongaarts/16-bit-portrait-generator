from __future__ import annotations

import asyncio
import io
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.config import settings
from app.palettes.loader import get_palette_colors, list_palettes

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle handler."""
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    # Try to init MediaPipe (works on CPU)
    try:
        from app.pipeline.face_detect import init_landmarker

        model_path = os.path.join(
            settings.model_cache_dir, "face_landmarker_v2_with_blendshapes.task"
        )
        if os.path.exists(model_path):
            init_landmarker(model_path)
            logger.info("MediaPipe FaceLandmarker ready")
    except Exception as e:
        logger.warning("MediaPipe init failed: %s", e)

    logger.info("16-Bit Portrait Generator ready (algorithmic pipeline)")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="16-Bit Portrait Generator",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS
origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Concurrency control
_semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

# Valid parameter values
VALID_OUTPUT_SIZES = {32, 48, 64, 96, 128}
VALID_PALETTE_COLORS = {8, 16, 24, 32, 64}
VALID_DITHER_MODES = {"none", "bayer2x2", "bayer4x4", "floyd-steinberg"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "pipeline": "algorithmic",
        "version": "0.2.0",
    }


@app.get("/api/palettes")
async def palettes():
    return {"palettes": list_palettes()}


@app.post("/api/generate")
async def generate(
    image: UploadFile = File(...),
    output_size: int = Form(64),
    palette_colors: int = Form(16),
    palette_name: Optional[str] = Form(None),
    dither: str = Form("none"),
    snes_snap: bool = Form(True),
    scale: int = Form(8),
    remove_bg: bool = Form(False),
    bg_threshold: float = Form(0.5),
):
    # Validate parameters
    if output_size not in VALID_OUTPUT_SIZES:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_parameter",
                "message": f"output_size must be one of {sorted(VALID_OUTPUT_SIZES)}",
            },
        )

    if palette_colors not in VALID_PALETTE_COLORS:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_parameter",
                "message": f"palette_colors must be one of {sorted(VALID_PALETTE_COLORS)}",
            },
        )

    if dither not in VALID_DITHER_MODES:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_parameter",
                "message": f"dither must be one of {sorted(VALID_DITHER_MODES)}",
            },
        )

    if scale < 1 or scale > 16:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_parameter",
                "message": "scale must be between 1 and 16",
            },
        )

    if not 0.0 <= bg_threshold <= 1.0:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_parameter",
                "message": "bg_threshold must be between 0.0 and 1.0",
            },
        )

    # Validate content type
    content_type = image.content_type or ""
    if not any(
        t in content_type for t in ("image/jpeg", "image/png", "image/webp", "octet-stream")
    ):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_format",
                "message": "Unsupported image format. Use JPEG, PNG, or WebP.",
            },
        )

    # Read image data
    image_data = await image.read()
    if len(image_data) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "image_too_large",
                "message": "Image exceeds 10MB limit.",
            },
        )

    # Decode image
    try:
        pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_array = np.array(pil_image)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_format",
                "message": "Could not decode image. Ensure it is a valid JPEG, PNG, or WebP.",
            },
        )

    # Resolve palette
    palette = None
    n_colors = palette_colors
    if palette_name:
        palette = get_palette_colors(palette_name, snes_snap=snes_snap)
        if palette is None:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "invalid_parameter",
                    "message": f"Unknown palette: {palette_name}",
                },
            )
        n_colors = len(palette)

    # Run pipeline
    try:
        async with _semaphore:
            from app.pipeline.fallback import run_algorithmic_pipeline

            start_time = time.time()
            output, palette_used, landmarks = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: run_algorithmic_pipeline(
                    input_array,
                    output_size=output_size,
                    n_colors=n_colors,
                    palette=palette,
                    dither=dither,
                    snes_snap=snes_snap,
                    scale=scale,
                    remove_bg=remove_bg,
                    bg_threshold=bg_threshold,
                ),
            )
            processing_ms = int((time.time() - start_time) * 1000)

    except Exception as e:
        if type(e).__name__ == "NoFaceDetectedError":
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "no_face_detected",
                    "message": "Could not detect a face in the uploaded image. "
                    "Ensure the photo contains a clearly visible face.",
                },
            )
        logger.error("Generation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "generation_failed",
                "message": f"Pipeline error: {str(e)}",
            },
        )

    # Encode output as PNG
    output_pil = Image.fromarray(output)
    buffer = io.BytesIO()
    output_pil.save(buffer, format="PNG")
    png_bytes = buffer.getvalue()

    # Build palette hex string
    palette_hex = ",".join(
        f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette_used
    )

    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={
            "X-Portrait-Palette": palette_hex,
            "X-Portrait-Size": str(output_size),
            "X-Portrait-Processing-Ms": str(processing_ms),
        },
    )


# Serve static test UI
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")
