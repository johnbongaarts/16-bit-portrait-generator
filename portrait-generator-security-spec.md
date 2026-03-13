# Portrait Generator: Security Headers for Iframe Embedding

## Context

The portrait generator is embedded as an iframe inside the WTFCRM app via `?mode=embed`. The CRM is deployed on Railway with a public domain. The portrait generator service must also be publicly accessible for the iframe to load in the user's browser, but it should be locked down so only the CRM can embed it.

## Requirements

### 1. Allow iframe embedding from the CRM domain only

Add response headers to the portrait generator so that only the CRM can embed it. This should apply to **all responses** served by the portrait generator.

**Headers to set:**

```
Content-Security-Policy: frame-ancestors https://YOUR_CRM_DOMAIN.railway.app
X-Frame-Options: ALLOW-FROM https://YOUR_CRM_DOMAIN.railway.app
```

- Replace `YOUR_CRM_DOMAIN.railway.app` with the actual CRM staging/production domain(s).
- If you have multiple CRM domains (staging + production), list them both in `frame-ancestors`:
  ```
  Content-Security-Policy: frame-ancestors https://staging.example.com https://production.example.com
  ```
- `X-Frame-Options` only supports one origin and is deprecated in favor of CSP `frame-ancestors`, but include it for older browser compatibility.

### 2. Make the allowed origin configurable via environment variable

Don't hardcode the CRM domain. Use an environment variable:

```
ALLOWED_FRAME_ANCESTORS=https://staging-crm.railway.app https://production-crm.railway.app
```

Then build the header dynamically:

```
Content-Security-Policy: frame-ancestors ${ALLOWED_FRAME_ANCESTORS}
```

**For local development**, default to `http://localhost:3000` if the env var is not set.

### 3. CORS headers (if the service serves API endpoints)

If the portrait generator also exposes API endpoints (not just the HTML page), add CORS headers:

```
Access-Control-Allow-Origin: https://YOUR_CRM_DOMAIN.railway.app
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Content-Type
```

This is only needed if the CRM makes `fetch()` calls directly to the portrait generator. Currently it does **not** — all communication is via `postMessage`. But it's good practice to restrict CORS anyway.

### 4. Deployment

- The portrait generator **must** have a public Railway domain assigned (Settings > Networking > Public Networking)
- Set the `ALLOWED_FRAME_ANCESTORS` env var in Railway to the CRM's public domain(s)
- On the CRM side, set `NEXT_PUBLIC_PORTRAIT_GENERATOR_URL` to the portrait generator's **public** Railway URL (not the `*.railway.internal` private URL)

## postMessage Protocol Reference

The CRM expects these messages from the portrait generator iframe:

| Message | Payload | Effect |
|---------|---------|--------|
| `{ type: 'portrait-complete', data: '<base64-png-data-url>' }` | Base64 PNG starting with `data:image/png;base64,` (max 50KB) | Saves portrait to CRM database, closes modal |
| `{ type: 'portrait-cancel' }` | None | Closes the modal without saving |

The iframe is loaded with `?mode=embed` query parameter.

## Verification

After deploying:
1. Open the portrait generator's public URL directly in a browser — it should load normally
2. Open the CRM, click `[PX]` on a contact to open the portrait editor — the iframe should load inside the CRT screen
3. Open browser dev tools Network tab — verify the portrait generator's response includes the `Content-Security-Policy: frame-ancestors ...` header
4. Try loading the portrait generator in an iframe from a different domain — it should be blocked
