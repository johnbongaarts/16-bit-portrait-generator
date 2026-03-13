# 16-Bit Portrait Generator

A Docker-deployed microservice that converts photographs into SNES-era (Chrono Trigger style) pixel art portraits. Exposes a REST API and includes a browser-based test UI.

## Two Processing Tiers

| Tier | Where | Speed | Quality | Cost |
|------|-------|-------|---------|------|
| **Browser** | Client-side JS | ~200ms | ~70% | Free |
| **Server** | GPU (SDXL + ControlNet) | 5–15s | ~85–90% | GPU required |

Both target **64×64 output at 16–32 colors** in the SNES 15-bit color space.

## Quick Start

### With Docker (GPU)

```bash
docker compose up --build
```

Service available at `http://localhost:8100`. Open the browser to see the test UI.

### Without GPU (CPU-only mode)

```bash
CPU_ONLY=true docker compose up --build
```

The server pipeline returns `503 gpu_unavailable` and the test UI falls back to the browser-tier pipeline.

### Local Development

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### `POST /api/generate`

Convert a photo to pixel art.

```bash
curl -X POST http://localhost:8100/api/generate \
  -F "image=@photo.jpg" \
  -F "output_size=64" \
  -F "palette_colors=16" \
  -o portrait.png
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `image` | file | required | JPEG/PNG/WebP, max 10MB |
| `output_size` | int | 64 | 32, 48, 64, 96, or 128 |
| `palette_colors` | int | 16 | 8, 16, 24, 32, or 64 |
| `palette_name` | string | null | Palette slug (overrides palette_colors) |
| `dither` | string | "none" | none, bayer2x2, bayer4x4, floyd-steinberg |
| `strength` | float | 0.7 | Denoising strength (0.0–1.0) |
| `style_prompt` | string | "" | Additional style keywords |
| `snes_snap` | bool | true | Snap to SNES 15-bit color space |
| `remove_bg` | bool | false | Remove background via GrabCut |
| `bg_threshold` | float | 0.5 | Background removal sensitivity (0.0–1.0) |
| `scale` | int | 8 | Upscale factor for output |

**Response:** PNG image with custom headers:
- `X-Portrait-Palette` — hex colors used
- `X-Portrait-Size` — output pixel size
- `X-Portrait-Processing-Ms` — processing time
- `X-Portrait-Fallback` — fallback mode if applicable

### `GET /api/palettes`

List available palettes.

### `GET /api/health`

Health check with GPU status.

## Built-in Palettes

| Name | Colors | Style |
|------|--------|-------|
| Chrono Trigger (Extracted) | 48 | Authentic SNES |
| Final Fantasy VI Portraits | 32 | Authentic SNES |
| Jehkoba32 | 32 | Warm JRPG |
| SNES Warm | 24 | Warm tones |
| SNES Cool | 24 | Cool tones |

## Test UI Features

### Export & Download

- **Output Size** controls the pixel-art grid resolution (e.g. 64×64). **Export Scale** controls the nearest-neighbor upscale factor for the final PNG (4×, 8×, 12×, or 16×). A 64px portrait at 8× exports as 512×512.
- **Export dimensions** are shown in the meta bar below the canvas (e.g. "Export: 512 × 512").
- **Changing Export Scale preserves retouches** — the canvas is rescaled in place without re-running the pipeline. All undo/redo snapshots are rescaled to match.
- **Download PNG** saves the current canvas at the selected export scale with a descriptive filename (e.g. `portrait-64px-8x.png`).
- **All Sizes** downloads four PNGs at 4×, 8×, 12×, and 16× in a single action, each with its own descriptive filename.

### Crop Adjustment

After uploading a photo, the face detection crop region is shown as a rotated square overlay on the source preview. Two sliders let you fine-tune the crop before (or after) generating:

- **Rotation** — adjust from -45° to +45° relative to the detected eye alignment angle
- **Crop Size** — scale the crop region from 50% to 200% of the detected face bounding box
- **Reset Crop** — return both sliders to their defaults

Adjustments re-render the portrait in real time. The crop override bypasses re-running face detection, so adjustments are instant.

### Portrait Viewer

- **Zoom** — scroll wheel zooms centered on the cursor, or use the +/- buttons and slider in the toolbar
- **Invert scroll** — checkbox to reverse scroll wheel zoom direction
- **Reset** — returns to 100% zoom
- Portrait stats (processing time, color count, export dimensions) are shown in a fixed toolbar above the canvas

### Retouch Tools

- **Pixel brush** (sizes 1-4) — left-click to paint, click+drag for strokes
- **Flood fill** — right-click to fill connected regions of the same color
- **Eyedropper** — Ctrl+click or toggle button to pick colors from the canvas
- **Undo/Redo** — up to 50 history states (Ctrl+Z / Ctrl+Shift+Z)

## Pipeline Architecture

### Server (GPU)
```
Photo → Face Detect → Identity Extract → SDXL img2img → Downscale → Quantize → Post-process → PNG
```

### Browser (JS)
```
Photo → Face Detect → Crop/Align → Pre-process → Downscale → Quantize → Post-process → Canvas
```

### Fallback Chain
1. IP-Adapter fails → generate without face identity
2. ControlNet fails → LoRA + IP-Adapter only
3. SDXL fails → algorithmic pipeline (same as browser tier)

## GPU Requirements

- NVIDIA GPU with 12+ GB VRAM
- CUDA 12.1+ with NVIDIA Container Toolkit
- ~10–12 GB VRAM at runtime

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Server port |
| `MODEL_CACHE_DIR` | /app/models | Model storage path |
| `MAX_CONCURRENT_REQUESTS` | 2 | GPU concurrency limit |
| `CPU_ONLY` | false | Disable GPU pipeline |
| `CORS_ORIGINS` | * | Allowed CORS origins |
| `LOG_LEVEL` | info | Logging level |

## Project Structure

```
portrait-generator/
├── app/
│   ├── main.py              # FastAPI app + routes
│   ├── config.py            # Settings from env vars
│   ├── pipeline/            # Processing pipeline modules
│   │   ├── face_detect.py   # MediaPipe face detection
│   │   ├── identity.py      # InsightFace embeddings
│   │   ├── generate.py      # SDXL generation
│   │   ├── downscale.py     # Nearest-neighbor downscale
│   │   ├── quantize.py      # OkLab k-means quantization
│   │   ├── postprocess.py   # Eye highlights, cleanup, upscale
│   │   ├── fallback.py      # Algorithmic pipeline
│   │   └── orchestrator.py  # Pipeline orchestration
│   ├── palettes/            # Color palette system
│   │   ├── loader.py        # Palette JSON loading
│   │   ├── snes_snap.py     # 15-bit color snapping
│   │   └── data/            # Palette JSON files
│   └── models/
│       └── loader.py        # Model initialization
├── static/
│   └── index.html           # Test UI (single file)
├── scripts/
│   └── download_models.py   # Model weight downloader
├── tests/                   # Unit + integration tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```
