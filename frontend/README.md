Chronora Frontend (React + Vite)

Quickstart

1. Install node and npm (Node 18+ recommended).
2. From `frontend/` install dependencies:

```bash
cd frontend
npm install
```

3. Run dev server:

```bash
npm run dev
```

4. Open the app at the URL printed by Vite (default: http://localhost:5173). The frontend expects the API at the same origin `/api/v1/*`. If the API runs on a different host:port, either enable CORS on the FastAPI server or change the fetch URLs in `src/components/GenerateForm.jsx`.

Notes
- Example prompts for architecture are available as quick buttons.
- The frontend posts a multipart/form-data request to `/api/v1/generate` and polls `/api/v1/job/{job_id}` for results.
- You may need to enable CORS in `sdxl_direct.py` for cross-origin dev: install `pip install fastapi[all]` and add `from fastapi.middleware.cors import CORSMiddleware` and `app.add_middleware(...)`.
