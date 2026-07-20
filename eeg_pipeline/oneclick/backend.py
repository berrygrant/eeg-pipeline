from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from eeg_pipeline import __version__
from eeg_pipeline.config import config_get, load_config
from eeg_pipeline.inputs import discover_pipeline_recordings

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class PipelineJob:
    id: str
    command: list[str]
    cwd: Path
    status: str = "queued"
    returncode: int | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    logs: list[str] = field(default_factory=list)
    error: str = ""
    process: subprocess.Popen[str] | None = None


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, PipelineJob] = {}
        self._lock = threading.Lock()

    def add(self, job: PipelineJob) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def get(self, job_id: str) -> PipelineJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[PipelineJob]:
        with self._lock:
            return list(self._jobs.values())

    def append_log(self, job_id: str, line: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.logs.append(line)
            if len(job.logs) > 2000:
                job.logs = job.logs[-2000:]

    def update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                setattr(job, key, value)


JOBS = JobStore()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _ok(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, default=_json_default).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Headers", "content-type")
    handler.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("content-length", "0") or "0")
    if length == 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object.")
    return data


def _config_path_from_payload(payload: dict[str, Any]) -> Path:
    value = payload.get("configPath")
    if not value:
        raise ValueError("configPath is required.")
    filename = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    allowed_configs = {
        "config.yaml": REPO_ROOT / "config.yaml",
        "config.yml": REPO_ROOT / "config.yml",
        "config.json": REPO_ROOT / "config.json",
    }
    if filename not in allowed_configs:
        raise ValueError("configPath must be config.yaml, config.yml, or config.json in this repository.")
    return allowed_configs[filename]


def _summarize_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": cfg.get("task"),
        "inputMode": config_get(cfg, "input.mode"),
        "bidsRoot": config_get(cfg, "paths.bids_root"),
        "rawDir": config_get(cfg, "paths.raw_dir"),
        "derivativesRoot": config_get(cfg, "paths.derivatives_root"),
        "subjects": config_get(cfg, "bids.subjects"),
        "sessions": config_get(cfg, "bids.sessions"),
        "tasks": config_get(cfg, "bids.tasks"),
        "runs": config_get(cfg, "bids.runs"),
        "erpEnabled": config_get(cfg, "metrics.erp.enabled"),
        "tfrEnabled": config_get(cfg, "metrics.tfr.enabled"),
        "icaMode": config_get(cfg, "ica.mode"),
        "useGpu": config_get(cfg, "compute.use_gpu"),
    }


def validate_config(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    warnings: list[str] = []
    derivatives_root = config_get(cfg, "paths.derivatives_root")
    if derivatives_root is None:
        warnings.append("No derivatives_root is set; the CLI will infer one from the input dataset.")
    if config_get(cfg, "metrics.tfr.enabled") and not config_get(cfg, "metrics.erp.enabled"):
        warnings.append("TFR is enabled while ERP metrics are disabled.")
    return {"ok": True, "configPath": config_path, "summary": _summarize_config(cfg), "warnings": warnings}


def discover_recordings(config_path: Path) -> dict[str, Any]:
    cfg = load_config(config_path)
    input_mode = str(config_get(cfg, "input.mode", "bids")).lower()
    recordings = discover_pipeline_recordings(
        mode=input_mode,
        bids_root=config_get(cfg, "paths.bids_root"),
        raw_dir=config_get(cfg, "paths.raw_dir"),
        subject_csv_dir=config_get(cfg, "paths.subject_csv_dir"),
        subjects=config_get(cfg, "bids.subjects"),
        sessions=config_get(cfg, "bids.sessions"),
        tasks=config_get(cfg, "bids.tasks"),
        runs=config_get(cfg, "bids.runs"),
        task_label=cfg.get("task"),
    )
    return {
        "ok": True,
        "count": len(recordings),
        "recordings": [
            {
                "subject": item.subject_label,
                "session": item.session_label,
                "task": item.task_id,
                "run": item.run_id,
                "sourceType": item.source_type,
                "rawPath": item.raw_path,
                "behaviorPath": item.behavior_path,
                "behaviorKind": item.behavior_kind,
            }
            for item in recordings
        ],
    }


def _stage_args(stages: dict[str, Any] | None) -> list[str]:
    if not stages:
        return ["--process_data", "--get_metrics"]
    args: list[str] = []
    if stages.get("processData", True):
        args.append("--process_data")
    if stages.get("getMetrics", True):
        args.append("--get_metrics")
    if stages.get("plotFigures", False):
        args.append("--plot_figures")
    if not args:
        raise ValueError("At least one stage must be selected.")
    return args


def _start_job(payload: dict[str, Any]) -> dict[str, Any]:
    config_path = _config_path_from_payload(payload)
    load_config(config_path)
    command = [
        sys.executable,
        "-m",
        "eeg_pipeline.cli",
        "--config",
        str(config_path),
        *_stage_args(payload.get("stages")),
    ]
    if payload.get("erpCore"):
        command.append("--erp-core")
    if payload.get("useGpu"):
        command.append("--use_gpu")

    job = PipelineJob(id=uuid.uuid4().hex, command=command, cwd=REPO_ROOT)
    JOBS.add(job)
    thread = threading.Thread(target=_run_job, args=(job.id,), daemon=True)
    thread.start()
    return {"ok": True, "job": _job_payload(job, include_logs=True)}


def _run_job(job_id: str) -> None:
    job = JOBS.get(job_id)
    if job is None:
        return
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            job.command,
            cwd=str(job.cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        JOBS.update(job_id, status="running", process=process)
        assert process.stdout is not None
        for line in process.stdout:
            JOBS.append_log(job_id, line.rstrip("\n"))
        returncode = process.wait()
        JOBS.update(
            job_id,
            status="succeeded" if returncode == 0 else "failed",
            returncode=returncode,
            finished_at=time.time(),
            process=None,
        )
    except Exception as exc:
        JOBS.append_log(job_id, f"[backend] {exc}")
        JOBS.update(job_id, status="failed", error=str(exc), finished_at=time.time(), process=None)


def _job_payload(job: PipelineJob, *, include_logs: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": job.id,
        "status": job.status,
        "returncode": job.returncode,
        "startedAt": job.started_at,
        "finishedAt": job.finished_at,
        "command": job.command,
        "error": job.error,
    }
    if include_logs:
        payload["logs"] = job.logs
    return payload


_LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _origin_allowed(handler: BaseHTTPRequestHandler) -> bool:
    """Reject browser cross-origin POSTs (CSRF) to this local command runner.

    A missing Origin (non-browser clients such as curl or the test suite) or the
    file:// renderer's ``null`` origin is allowed; any real, non-loopback web
    origin is rejected before a pipeline job can be started. The server binds to
    127.0.0.1 with a permissive CORS header, so without this check any website
    open in the user's browser could POST /api/run.
    """
    origin = handler.headers.get("Origin")
    if not origin or origin == "null":
        return True
    try:
        host = urlparse(origin).hostname
    except Exception:
        return False
    return host in _LOCAL_HOSTS


class OneClickHandler(BaseHTTPRequestHandler):
    server_version = "EEGPipelineOneClick/0.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def do_OPTIONS(self) -> None:
        _ok(self, {"ok": True})

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/health":
                _ok(self, {"ok": True, "version": __version__, "repoRoot": REPO_ROOT})
                return
            if parsed.path == "/api/jobs":
                _ok(self, {"ok": True, "jobs": [_job_payload(job, include_logs=False) for job in JOBS.list()]})
                return
            if parsed.path.startswith("/api/jobs/"):
                job_id = parsed.path.rsplit("/", 1)[-1]
                job = JOBS.get(job_id)
                if job is None:
                    _ok(self, {"ok": False, "error": "Job not found."}, status=404)
                    return
                _ok(self, {"ok": True, "job": _job_payload(job, include_logs=True)})
                return
            _ok(self, {"ok": False, "error": "Not found."}, status=404)
        except Exception as exc:
            _ok(self, {"ok": False, "error": str(exc)}, status=500)

    def do_POST(self) -> None:
        if not _origin_allowed(self):
            _ok(self, {"ok": False, "error": "Cross-origin request rejected."}, status=403)
            return
        try:
            payload = _read_json(self)
            if self.path == "/api/config/validate":
                _ok(self, validate_config(_config_path_from_payload(payload)))
                return
            if self.path == "/api/recordings/discover":
                _ok(self, discover_recordings(_config_path_from_payload(payload)))
                return
            if self.path == "/api/run":
                _ok(self, _start_job(payload), status=202)
                return
            _ok(self, {"ok": False, "error": "Not found."}, status=404)
        except Exception as exc:
            _ok(self, {"ok": False, "error": str(exc)}, status=400)


def _find_port(host: str, preferred_port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, preferred_port))
            return preferred_port
        except OSError:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])


def serve(host: str = "127.0.0.1", port: int = 8765) -> None:
    actual_port = _find_port(host, port)
    server = ThreadingHTTPServer((host, actual_port), OneClickHandler)
    print(json.dumps({"event": "ready", "host": host, "port": actual_port}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the eeg-pipeline one-click backend.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(host=args.host, port=args.port)


if __name__ == "__main__":  # pragma: no cover
    main()
