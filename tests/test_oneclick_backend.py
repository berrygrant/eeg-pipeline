import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib import request

import pytest

from eeg_pipeline.oneclick import backend


def test_stage_args_defaults_to_process_and_metrics():
    assert backend._stage_args(None) == ["--process_data", "--get_metrics"]


def test_stage_args_uses_selected_stages():
    assert backend._stage_args({"processData": False, "getMetrics": True, "plotFigures": True}) == [
        "--get_metrics",
        "--plot_figures",
    ]


def test_stage_args_rejects_empty_selection():
    with pytest.raises(ValueError, match="At least one stage"):
        backend._stage_args({"processData": False, "getMetrics": False, "plotFigures": False})


def test_config_path_from_payload_maps_allowed_basename_to_repo(tmp_path: Path):
    outside_config = tmp_path / "config.yaml"
    outside_config.write_text("{}", encoding="utf-8")

    assert backend._config_path_from_payload({"configPath": str(outside_config)}) == backend.REPO_ROOT / "config.yaml"


def test_config_path_from_payload_rejects_unapproved_names():
    with pytest.raises(ValueError, match="config.yaml"):
        backend._config_path_from_payload({"configPath": "tmp/custom.yaml"})


def test_validate_config_reports_summary(tmp_path: Path):
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
input:
  mode: bids
paths:
  bids_root: {bids_root}
events:
  behavioral_keep_codes: [1, 2]
  standard_codes: [1]
  deviant_codes: [2]
""",
        encoding="utf-8",
    )

    result = backend.validate_config(cfg_path)

    assert result["ok"] is True
    assert result["summary"]["inputMode"] == "bids"
    assert result["summary"]["bidsRoot"] == bids_root


def test_job_store_tracks_updates_and_logs():
    store = backend.JobStore()
    job = backend.PipelineJob(id="job-1", command=["python"], cwd=backend.REPO_ROOT)

    store.add(job)
    store.append_log("job-1", "line")
    store.update("job-1", status="running", returncode=0)

    assert store.get("job-1") is job
    assert store.list() == [job]
    assert job.logs == ["line"]
    assert job.status == "running"
    assert job.returncode == 0


def test_job_store_ignores_missing_jobs_and_trims_logs():
    store = backend.JobStore()
    job = backend.PipelineJob(id="job-1", command=["python"], cwd=backend.REPO_ROOT)
    store.add(job)

    store.append_log("missing", "ignored")
    store.update("missing", status="ignored")
    for idx in range(2002):
        store.append_log("job-1", str(idx))

    assert len(job.logs) == 2000
    assert job.logs[0] == "2"
    assert job.status == "queued"


class _FakeHandler:
    def __init__(self, payload=None):
        raw = b"" if payload is None else json.dumps(payload).encode("utf-8")
        self.headers = {"content-length": str(len(raw))}
        self.rfile = io.BytesIO(raw)
        self.wfile = io.BytesIO()
        self.status = None
        self.response_headers = []
        self.ended = False

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.response_headers.append((key, value))

    def end_headers(self):
        self.ended = True


def test_http_json_helpers_round_trip_payload():
    handler = _FakeHandler({"configPath": "config.yaml"})

    assert backend._read_json(handler) == {"configPath": "config.yaml"}

    backend._ok(handler, {"ok": True, "path": backend.REPO_ROOT})
    body = json.loads(handler.wfile.getvalue().decode("utf-8"))

    assert handler.status == 200
    assert handler.ended is True
    assert body["ok"] is True
    assert body["path"] == str(backend.REPO_ROOT)


def test_http_json_helpers_handle_empty_and_invalid_payloads():
    assert backend._read_json(_FakeHandler()) == {}
    with pytest.raises(ValueError, match="JSON object"):
        backend._read_json(_FakeHandler(["not", "a", "mapping"]))
    with pytest.raises(TypeError, match="not JSON serializable"):
        backend._json_default(object())


def test_config_path_from_payload_requires_value():
    with pytest.raises(ValueError, match="required"):
        backend._config_path_from_payload({})


def test_discover_recordings_serializes_recording_metadata(monkeypatch):
    recording = SimpleNamespace(
        subject_label="sub-01",
        session_label=None,
        task_id="oddball",
        run_id="01",
        source_type="bids",
        raw_path=Path("/tmp/raw.vhdr"),
        behavior_path=Path("/tmp/events.tsv"),
        behavior_kind="bids_events",
    )
    monkeypatch.setattr(
        backend,
        "load_config",
        lambda path: {
            "task": "oddball",
            "input": {"mode": "bids"},
            "paths": {"bids_root": Path("/tmp/bids"), "raw_dir": None, "subject_csv_dir": None},
            "bids": {"subjects": None, "sessions": None, "tasks": None, "runs": None},
        },
    )
    monkeypatch.setattr(backend, "discover_pipeline_recordings", lambda **kwargs: [recording])

    result = backend.discover_recordings(Path("config.yaml"))

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["recordings"][0]["subject"] == "sub-01"


def test_validate_config_reports_tfr_without_erp_warning(tmp_path: Path):
    bids_root = tmp_path / "bids"
    bids_root.mkdir()
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        f"""
input:
  mode: bids
paths:
  bids_root: {bids_root}
events:
  behavioral_keep_codes: [1, 2]
  standard_codes: [1]
  deviant_codes: [2]
metrics:
  erp:
    enabled: false
  tfr:
    enabled: true
""",
        encoding="utf-8",
    )

    result = backend.validate_config(cfg_path)

    assert any("TFR is enabled" in warning for warning in result["warnings"])


def test_start_job_builds_cli_command_without_running_thread(monkeypatch):
    started = []

    class FakeThread:
        def __init__(self, target, args, daemon):
            self.target = target
            self.args = args
            self.daemon = daemon

        def start(self):
            started.append(self.args)

    monkeypatch.setattr(backend, "load_config", lambda path: {})
    monkeypatch.setattr(backend.threading, "Thread", FakeThread)

    result = backend._start_job(
        {
            "configPath": "config.yaml",
            "stages": {"processData": True, "getMetrics": False, "plotFigures": True},
            "erpCore": True,
            "useGpu": True,
        }
    )

    command = result["job"]["command"]
    assert result["ok"] is True
    assert started == [(result["job"]["id"],)]
    assert command[:3] == [backend.sys.executable, "-m", "eeg_pipeline.cli"]
    assert "--process_data" in command
    assert "--plot_figures" in command
    assert "--get_metrics" not in command
    assert "--erp-core" in command
    assert "--use_gpu" in command


def test_run_job_captures_stdout_and_success(monkeypatch):
    store = backend.JobStore()
    monkeypatch.setattr(backend, "JOBS", store)
    job = backend.PipelineJob(id="job-1", command=["fake"], cwd=backend.REPO_ROOT)
    store.add(job)

    class FakeProcess:
        stdout = ["first\n", "second\n"]

        def wait(self):
            return 0

    monkeypatch.setattr(backend.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())

    backend._run_job("job-1")

    assert job.status == "succeeded"
    assert job.returncode == 0
    assert job.logs == ["first", "second"]
    assert job.finished_at is not None


def test_run_job_handles_missing_job_and_process_failure(monkeypatch):
    store = backend.JobStore()
    monkeypatch.setattr(backend, "JOBS", store)

    backend._run_job("missing")

    job = backend.PipelineJob(id="job-1", command=["fake"], cwd=backend.REPO_ROOT)
    store.add(job)
    monkeypatch.setattr(backend.subprocess, "Popen", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    backend._run_job("job-1")

    assert job.status == "failed"
    assert job.error == "boom"
    assert job.logs == ["[backend] boom"]


def test_handler_serves_health_and_post_endpoint(monkeypatch):
    monkeypatch.setattr(
        backend,
        "validate_config",
        lambda path: {"ok": True, "configPath": path, "summary": {"inputMode": "bids"}, "warnings": []},
    )
    server = backend.ThreadingHTTPServer(("127.0.0.1", 0), backend.OneClickHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        with request.urlopen(f"{base_url}/api/health", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))
        post = request.Request(
            f"{base_url}/api/config/validate",
            data=json.dumps({"configPath": "config.yaml"}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with request.urlopen(post, timeout=5) as response:
            validation = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert health["ok"] is True
    assert validation["ok"] is True
    assert validation["summary"]["inputMode"] == "bids"


def test_handler_serves_jobs_and_error_paths(monkeypatch):
    store = backend.JobStore()
    job = backend.PipelineJob(id="job-1", command=["python"], cwd=backend.REPO_ROOT, logs=["ok"])
    store.add(job)
    monkeypatch.setattr(backend, "JOBS", store)
    monkeypatch.setattr(backend, "discover_recordings", lambda path: {"ok": True, "count": 0, "recordings": []})
    monkeypatch.setattr(backend, "_start_job", lambda payload: {"ok": True, "job": {"id": "new-job"}})
    server = backend.ThreadingHTTPServer(("127.0.0.1", 0), backend.OneClickHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        options = request.Request(f"{base_url}/api/health", method="OPTIONS")
        with request.urlopen(options, timeout=5) as response:
            options_payload = json.loads(response.read().decode("utf-8"))
        with request.urlopen(f"{base_url}/api/jobs", timeout=5) as response:
            jobs = json.loads(response.read().decode("utf-8"))
        with request.urlopen(f"{base_url}/api/jobs/job-1", timeout=5) as response:
            job_payload = json.loads(response.read().decode("utf-8"))
        with pytest.raises(Exception):
            request.urlopen(f"{base_url}/api/jobs/missing", timeout=5)
        with pytest.raises(Exception):
            request.urlopen(f"{base_url}/api/unknown", timeout=5)
        discover_post = request.Request(
            f"{base_url}/api/recordings/discover",
            data=json.dumps({"configPath": "config.yaml"}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with request.urlopen(discover_post, timeout=5) as response:
            discovery = json.loads(response.read().decode("utf-8"))
        run_post = request.Request(
            f"{base_url}/api/run",
            data=json.dumps({"configPath": "config.yaml"}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with request.urlopen(run_post, timeout=5) as response:
            run = json.loads(response.read().decode("utf-8"))
        post = request.Request(
            f"{base_url}/api/unknown",
            data=json.dumps({"configPath": "config.yaml"}).encode("utf-8"),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with pytest.raises(Exception):
            request.urlopen(post, timeout=5)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert options_payload["ok"] is True
    assert jobs["jobs"][0]["id"] == "job-1"
    assert job_payload["job"]["logs"] == ["ok"]
    assert discovery["count"] == 0
    assert run["job"]["id"] == "new-job"


def test_find_port_falls_back_when_preferred_port_is_busy():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        busy_port = sock.getsockname()[1]

        selected = backend._find_port("127.0.0.1", busy_port)

    assert selected != busy_port


def test_find_port_returns_available_preferred_port():
    assert backend._find_port("127.0.0.1", 0) == 0


def test_main_passes_cli_args_to_serve(monkeypatch):
    captured = {}

    def fake_serve(*, host, port):
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr(backend, "serve", fake_serve)

    backend.main(["--host", "127.0.0.2", "--port", "9876"])

    assert captured == {"host": "127.0.0.2", "port": 9876}
