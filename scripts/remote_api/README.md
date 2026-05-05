# Remote Inference API

This folder contains a small FastAPI service that runs inference jobs on a remote server and writes logs/results into a local jobs directory.

## Files
- `server.py`: FastAPI server (`/infer`, `/status/{job_id}`, `/logs/{job_id}`, `/result/{job_id}`).
- `sample_infer.py`: Sample stub that writes a dummy result.
- `config.example.json`: Example configuration (copy to `config.json`).
- `requirements.txt`: Python dependencies.

## Quick Start (Remote)
1. Copy this folder to the remote server.
2. Create `scripts/remote_api/config.json` based on `config.example.json`.
3. Install deps:
   ```bash
   pip install -r scripts/remote_api/requirements.txt
   ```
4. Run server:
   ```bash
   python scripts/remote_api/server.py --config scripts/remote_api/config.json
   ```

## API
- `POST /infer` JSON: `{ "text": "...", "job_id": "optional", "meta": { ... } }`
- `GET /status/{job_id}`
- `GET /logs/{job_id}`
- `GET /result/{job_id}`
- `GET /download/{job_id}`

Results are written to `outputs/remote_inference/jobs/<job_id>/output/result.json`.
