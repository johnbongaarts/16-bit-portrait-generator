# Ethics Statement: 16-Bit Portrait Generator

## What This Document Is

An honest assessment of the ethical dimensions of this project. Not a legal shield,
not a marketing document. This is meant for contributors, users, and anyone evaluating
whether to deploy, fork, or use this tool.

---

## Architecture: Algorithmic Pipeline Only

This project uses a **purely algorithmic pipeline**. No generative AI models are used.

The pipeline applies deterministic operations to transform an input photo into pixel art:
MediaPipe face landmark detection, Gaussian blur, contrast/saturation adjustment,
area downscaling, k-means color quantization (in OkLab perceptual color space),
SNES 15-bit color snapping, optional dithering, and nearest-neighbor upscaling.

No image is generated — the input photo is transformed through a fixed sequence of
mathematical operations. The output is a direct derivative of the input, not a
synthesis from a learned distribution.

The only model involved is Google's MediaPipe FaceLandmarker (~3 MB), a lightweight
face geometry model that outputs 478 landmark coordinates. No biometric templates
or identity embeddings are extracted or stored.

### Why No Generative Pipeline?

An earlier version of this project included a GPU pipeline using Stable Diffusion XL,
InsightFace, ControlNet, IP-Adapter FaceID, and a pixel-art LoRA. That pipeline was
removed due to unresolvable ethical concerns:

- **SDXL** was trained on LAION-5B, a dataset scraped without consent from the open
  internet. CSAM was discovered in the dataset. Ongoing litigation from artists and
  Getty Images challenges the legality of the training process.
- **InsightFace** was trained on MS1MV2, a derivative of Microsoft's retracted
  MS-Celeb-1M dataset containing images of ~100,000 people scraped without consent.
  Using it constitutes biometric data processing under GDPR, BIPA, and similar laws.
- **The pixel-art LoRA** was trained on pixel art scraped from the web, reproducing
  a skilled craft without compensating or crediting the artists whose work was used.
- **IP-Adapter FaceID** enabled generating novel images of recognizable individuals,
  which is functionally adjacent to deepfake technology.

The decision was made to remove the generative pipeline entirely rather than ship
these unresolved issues.

---

## Remaining Ethical Considerations

### Face Detection Bias

MediaPipe is documented to have **accuracy disparities** across:
- Skin tone (lower accuracy on darker skin)
- Facial structure (optimized primarily on Western/East Asian features)
- Lighting conditions (underperforms on subjects with dark skin in low light)
- Age (lower accuracy on very young and very old faces)

The mirror-retry logic in `face_detect.py` (flipping horizontally for side profiles)
partially mitigates detection failures, but the underlying model biases remain.

This means some users will experience more "No face detected" errors than others,
correlated with their demographic characteristics.

### Consent and Privacy

- Photos are processed in memory and not persisted to disk by default.
- No biometric templates or identity embeddings are extracted. MediaPipe landmarks
  are geometric coordinates, not identity vectors.
- However, the codebase includes **no explicit data retention policy**, no privacy
  notice, and no consent mechanism.
- Users may upload photos of **other people** who have not consented to having their
  face processed.

### Background Removal

The optional background removal feature uses OpenCV GrabCut, a classical graph-cut
segmentation algorithm. This is a purely algorithmic operation with no learned model
component beyond the MediaPipe landmarks used to seed the initial mask.

---

## Recommendations for Deployment

1. **Add a privacy notice** explaining what happens to uploaded images (processed
   in memory, not stored, no biometric extraction).
2. **Add consent language** stating users should only upload photos of themselves
   or with the subject's consent.
3. **Implement logging and data retention policies** (what is logged, for how long).
4. **Test across demographics** — build a face detection test suite that includes
   diverse skin tones, ages, and facial structures. Document accuracy disparities.
5. **Add rate limiting and abuse prevention** before any public deployment.

---

## Summary of Ethical Risks

| Risk | Severity | Mitigable? |
|------|----------|------------|
| Face detection bias across demographics | Medium | Partially — model improvements, testing |
| No consent mechanism for photo subjects | Medium | Yes — add UI/policy |
| No data retention policy | Low | Yes — add policy and enforcement |

---

## What This Project Is and Isn't

**Is:**
- A purely algorithmic tool that transforms photos into pixel art through
  deterministic mathematical operations.
- A project that intentionally chose not to use generative AI due to ethical concerns.

**Is not:**
- A generative AI tool (no images are synthesized from a learned distribution).
- A biometric processing system (no identity embeddings are extracted).
- A production service (it has no auth, no rate limiting, no privacy policy).
