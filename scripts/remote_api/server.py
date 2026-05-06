import argparse
import json
import os
import threading
import time
import uuid
from http import HTTPStatus
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse


def load_config(config_path: str) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def format_command(template: list[str], input_file: Path, output_dir: Path, job_id: str) -> list[str]:
    return [
        part.format(input_file=input_file.as_posix(), output_dir=output_dir.as_posix(), job_id=job_id)
        for part in template
    ]


def write_status(job_dir: Path, status: str, detail: str = "") -> None:
    payload = {
        "status": status,
        "detail": detail,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (job_dir / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_job(job_dir: Path, command: list[str]) -> None:
    log_path = job_dir / "run.log"
    write_status(job_dir, "running")
    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("Command: " + " ".join(command) + "\n")
            log_file.flush()
            exit_code = os.system(" ".join(command) + f" >> \"{log_path}\" 2>&1")
        if exit_code == 0:
            write_status(job_dir, "completed")
        else:
            write_status(job_dir, "failed", f"exit_code={exit_code}")
    except Exception as exc:
        write_status(job_dir, "failed", str(exc))


def create_app(config: dict[str, Any]) -> FastAPI:
    app = FastAPI(title="Remote Inference API")

    jobs_root = Path(config["jobs_root"])
    ensure_dir(jobs_root)
    command_template = config["command"]

    @app.post("/infer")
    async def infer(payload: dict[str, Any]) -> JSONResponse:
        text = str(payload.get("text", "")).strip()
        if not text:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="text is required")

        meta = payload.get("meta", {}) or {}
        task = payload.get("task") or meta.get("task")
        if task:
            meta["task"] = str(task).strip().lower()

        job_id = payload.get("job_id") or uuid.uuid4().hex
        job_dir = jobs_root / job_id
        ensure_dir(job_dir)

        input_file = job_dir / "input.json"
        input_file.write_text(
            json.dumps({"text": text, "meta": meta}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        output_dir = job_dir / "output"
        ensure_dir(output_dir)

        command = format_command(command_template, input_file, output_dir, job_id)
        write_status(job_dir, "queued")

        thread = threading.Thread(target=run_job, args=(job_dir, command), daemon=True)
        thread.start()

        return JSONResponse({"job_id": job_id, "status": "queued"})

    @app.get("/status/{job_id}")
    async def status(job_id: str) -> JSONResponse:
        status_file = jobs_root / job_id / "status.json"
        if not status_file.exists():
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="job not found")
        payload = json.loads(status_file.read_text(encoding="utf-8"))
        return JSONResponse(payload)

    @app.get("/logs/{job_id}")
    async def logs(job_id: str) -> PlainTextResponse:
        log_path = jobs_root / job_id / "run.log"
        if not log_path.exists():
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="log not found")
        return PlainTextResponse(log_path.read_text(encoding="utf-8"))

    @app.get("/result/{job_id}")
    async def result(job_id: str) -> JSONResponse:
        result_path = jobs_root / job_id / "output" / "result.json"
        if not result_path.exists():
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="result not found")
        return JSONResponse(json.loads(result_path.read_text(encoding="utf-8")))

    @app.get("/download/{job_id}")
    async def download(job_id: str) -> FileResponse:
        result_path = jobs_root / job_id / "output" / "result.json"
        if not result_path.exists():
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="result not found")
        return FileResponse(result_path)

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Remote Inference API Server")
    parser.add_argument("--config", default="scripts/remote_api/config.json")
    args = parser.parse_args()

    config = load_config(args.config)
    app = create_app(config)

    import uvicorn

    uvicorn.run(app, host=config["listen_host"], port=int(config["listen_port"]))


if __name__ == "__main__":
    main()
