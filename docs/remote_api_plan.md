# Remote API + rsync plan

## Goal
Run Stage4 on the remote server, run Stage5 frontend locally, and sync inference logs/results back to local.

## Remote components
- `scripts/remote_api/server.py`: API server to accept text input and launch an inference job.
- `outputs/remote_inference/jobs/<job_id>`: job directory storing input, logs, and result.

## Local components
- `scripts/local_sync/rsync_pull.ps1` or `rsync_pull.sh`: pull remote job outputs.
- Stage5 API reads local `outputs/remote_inference/jobs` for display (manual integration later).

## Job layout
```
outputs/remote_inference/jobs/<job_id>/
  input.json
  status.json
  run.log
  output/
    result.json
```

## Workflow
1. Remote: start API server.
2. Local: send text to remote `/infer`.
3. Remote: job writes logs/results.
4. Local: run rsync script to pull results.
5. Local: read results in Stage5 UI (future wiring).
