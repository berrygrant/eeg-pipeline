import builtins
import sys
from types import SimpleNamespace

import numpy as np

import eeg_pipeline.gpu as gpu


def teardown_function():
    gpu.configure(False)


def test_configure_false_resets_to_numpy_backend():
    gpu._set_backend(xp="sentinel", name="cupy", enabled=True)

    status = gpu.configure(False, device=3)

    assert status == {
        "requested": False,
        "enabled": False,
        "backend": "numpy",
        "mne_cuda": "disabled",
        "cupy": "disabled",
        "device": 3,
    }
    assert gpu.backend() == "numpy"
    assert gpu.gpu_enabled() is False


def test_configure_true_uses_cupy_when_available(monkeypatch):
    fake_cp = SimpleNamespace()

    monkeypatch.setattr(gpu, "_try_init_mne_cuda", lambda device: "initialized")
    monkeypatch.setattr(gpu, "_try_init_cupy", lambda device: ("available", fake_cp))

    status = gpu.configure(True, device=1)

    assert status["enabled"] is True
    assert status["backend"] == "cupy"
    assert gpu.get_xp() is fake_cp
    assert gpu.backend() == "cupy"


def test_configure_true_falls_back_to_numpy_when_cupy_is_unavailable(monkeypatch):
    monkeypatch.setattr(gpu, "_try_init_mne_cuda", lambda device: "available")
    monkeypatch.setattr(gpu, "_try_init_cupy", lambda device: ("unavailable: no cupy", None))

    status = gpu.configure(True)

    assert status["requested"] is True
    assert status["enabled"] is True
    assert status["backend"] == "numpy"
    assert gpu.backend() == "numpy"


def test_capability_report_and_to_numpy_use_mocked_modules(monkeypatch):
    class FakeRuntime:
        @staticmethod
        def getDeviceCount():
            return 2

        @staticmethod
        def getDevice():
            return 1

        @staticmethod
        def getDeviceProperties(device):
            return {
                "name": b"Mock GPU",
                "major": 8,
                "minor": 6,
                "totalGlobalMem": 4 * (1024 ** 3),
            }

    class FakeDevice:
        def __init__(self, device):
            self.device = device

        def use(self):
            return None

    fake_cp = SimpleNamespace(
        __version__="1.2.3",
        cuda=SimpleNamespace(runtime=FakeRuntime(), Device=FakeDevice),
        asnumpy=lambda x: ["converted", x],
    )
    fake_mne = SimpleNamespace(__version__="9.9.9")

    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setitem(sys.modules, "mne", fake_mne)

    gpu._set_backend(fake_cp, "cupy", True)
    gpu._DEVICE = 0

    rep = gpu.capability_report()
    msg = gpu.format_capability_report(rep)

    assert rep["mne_version"] == "9.9.9"
    assert rep["cupy_version"] == "1.2.3"
    assert rep["gpu_count"] == 2
    assert rep["device"] == 0
    assert rep["device_name"] == "Mock GPU"
    assert rep["compute_capability"] == "8.6"
    assert rep["total_mem_gb"] == 4.0
    assert "gpu_count=2" in msg
    assert "name=Mock GPU" in msg
    assert gpu.to_numpy("x") == ["converted", "x"]

    gpu._set_backend(np, "numpy", False)
    assert np.array_equal(gpu.to_numpy([1, 2]), np.array([1, 2]))


def test_try_init_mne_cuda_covers_import_and_capability_paths(monkeypatch):
    original_import = builtins.__import__

    def missing_mne(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "mne":
            raise ImportError("no mne")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_mne)
    assert gpu._try_init_mne_cuda(None).startswith("mne_unavailable:")

    monkeypatch.setattr(builtins, "__import__", original_import)
    monkeypatch.setitem(sys.modules, "mne", SimpleNamespace())
    assert gpu._try_init_mne_cuda(None) == "mne_cuda_not_available"

    calls = {}
    cuda_mod = SimpleNamespace(
        set_cuda_device=lambda device: calls.setdefault("device", device),
    )
    monkeypatch.setitem(sys.modules, "mne", SimpleNamespace(cuda=cuda_mod))
    assert gpu._try_init_mne_cuda(2) == "available"
    assert calls["device"] == 2


def test_try_init_mne_cuda_covers_initialized_and_error_paths(monkeypatch):
    calls = {}
    cuda_ok = SimpleNamespace(
        set_cuda_device=lambda device: calls.setdefault("device", device),
        init_cuda=lambda: calls.setdefault("init", True),
    )
    monkeypatch.setitem(sys.modules, "mne", SimpleNamespace(cuda=cuda_ok))
    assert gpu._try_init_mne_cuda(1) == "initialized"
    assert calls == {"device": 1, "init": True}

    def boom():
        raise RuntimeError("cuda boom")

    cuda_bad = SimpleNamespace(init_cuda=boom)
    monkeypatch.setitem(sys.modules, "mne", SimpleNamespace(cuda=cuda_bad))
    assert gpu._try_init_mne_cuda(None) == "error: cuda boom"


def test_try_init_cupy_covers_available_and_import_failure(monkeypatch):
    class FakeDevice:
        def __init__(self, device):
            self.device = device

        def use(self):
            return None

    fake_cp = SimpleNamespace(cuda=SimpleNamespace(Device=FakeDevice))
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)

    status, xp = gpu._try_init_cupy(3)
    assert status == "available"
    assert xp is fake_cp

    status, xp = gpu._try_init_cupy(None)
    assert status == "available"
    assert xp is fake_cp

    original_import = builtins.__import__

    def missing_cupy(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cupy":
            raise ImportError("no cupy")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_cupy)
    status, xp = gpu._try_init_cupy(None)
    assert status.startswith("unavailable:")
    assert xp is None


def test_capability_report_covers_runtime_error_paths_and_empty_format(monkeypatch):
    class Runtime:
        @staticmethod
        def getDeviceCount():
            raise RuntimeError("count boom")

        @staticmethod
        def getDevice():
            raise RuntimeError("device boom")

        @staticmethod
        def getDeviceProperties(device):
            raise RuntimeError("props boom")

    fake_cp = SimpleNamespace(
        __version__="2.0.0",
        cuda=SimpleNamespace(runtime=Runtime()),
    )
    fake_mne = SimpleNamespace(__version__="1.0.0")

    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setitem(sys.modules, "mne", fake_mne)

    gpu._DEVICE = None
    rep = gpu.capability_report()

    assert rep["mne_version"] == "1.0.0"
    assert rep["cupy_version"] == "2.0.0"
    assert rep["gpu_count"] is None
    assert rep["device"] is None
    assert "device_info_error" not in rep
    assert gpu.format_capability_report({}) == ""


def test_capability_report_covers_device_property_errors_and_import_failures(monkeypatch):
    class Runtime:
        @staticmethod
        def getDeviceCount():
            return 1

        @staticmethod
        def getDevice():
            return 0

        @staticmethod
        def getDeviceProperties(device):
            raise RuntimeError("props boom")

    fake_cp = SimpleNamespace(
        __version__="2.0.0",
        cuda=SimpleNamespace(runtime=Runtime()),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setitem(sys.modules, "mne", SimpleNamespace(__version__="1.0.0"))

    gpu._DEVICE = 0
    rep = gpu.capability_report()
    msg = gpu.format_capability_report(rep)

    assert rep["device_info_error"] == "props boom"
    assert "device_info_error=props boom" in msg

    original_import = builtins.__import__

    def missing_modules(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"mne", "cupy"}:
            raise ImportError(f"missing {name}")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_modules)
    rep = gpu.capability_report()
    msg = gpu.format_capability_report(rep)

    assert rep["mne_version"].startswith("unavailable:")
    assert rep["cupy_error"].startswith("missing cupy")
    assert "cupy_error=missing cupy" in msg


def test_capability_report_covers_non_byte_device_name_and_partial_properties(monkeypatch):
    class Runtime:
        @staticmethod
        def getDeviceCount():
            return 1

        @staticmethod
        def getDevice():
            return 0

        @staticmethod
        def getDeviceProperties(device):
            return {
                "name": "Mock GPU",
                "major": 8,
                "minor": None,
            }

    fake_cp = SimpleNamespace(
        __version__="3.0.0",
        cuda=SimpleNamespace(runtime=Runtime()),
    )
    monkeypatch.setitem(sys.modules, "cupy", fake_cp)
    monkeypatch.setitem(sys.modules, "mne", SimpleNamespace(__version__="1.0.0"))

    gpu._DEVICE = None
    rep = gpu.capability_report()

    assert rep["device"] == 0
    assert rep["device_name"] == "Mock GPU"
    assert "compute_capability" not in rep
    assert "total_mem_gb" not in rep


def test_filter_n_jobs_passes_through_cpu_workers_when_gpu_disabled():
    gpu.configure(False)

    # No GPU: the requested CPU worker count must reach MNE unchanged.
    assert gpu.filter_n_jobs(1) == 1
    assert gpu.filter_n_jobs(8) == 8
    assert gpu.filter_n_jobs(-1) == -1


def test_filter_n_jobs_routes_to_cuda_when_mne_cuda_is_active(monkeypatch):
    # MNE only uses CUDA for FFT filtering when n_jobs == "cuda"; initializing
    # CUDA is not enough on its own. Without this routing --use_gpu had no
    # effect on filtering at all.
    monkeypatch.setattr(gpu, "_GPU_ENABLED", True)
    monkeypatch.setattr(gpu, "_MNE_CUDA_STATUS", "initialized")

    # Routing to CUDA also discards the CPU worker count: CUDA filtering is
    # single-stream, so n_jobs has no meaning once "cuda" is selected.
    assert gpu.filter_n_jobs(8) == "cuda"
    assert gpu.filter_n_jobs(1) == "cuda"


def test_filter_n_jobs_stays_on_cpu_when_cuda_init_failed(monkeypatch):
    # A cupy-only backend (or a failed init_cuda) must not claim CUDA filtering.
    monkeypatch.setattr(gpu, "_GPU_ENABLED", True)
    monkeypatch.setattr(gpu, "_MNE_CUDA_STATUS", "error: no device")

    assert gpu.filter_n_jobs(4) == 4


def test_filter_n_jobs_requires_initialized_cuda_not_merely_available(monkeypatch):
    """"available" means the mne.cuda module exists, not that CUDA initialized.

    Claiming CUDA there would hand MNE an n_jobs="cuda" it cannot honor.
    """
    monkeypatch.setattr(gpu, "_GPU_ENABLED", True)
    monkeypatch.setattr(gpu, "_MNE_CUDA_STATUS", "available")

    assert gpu.filter_n_jobs(4) == 4


def test_filter_n_jobs_requires_mne_to_report_actual_cuda_capability(monkeypatch):
    """"initialized" only means init_cuda() did not raise.

    MNE's init_cuda no-ops WITHOUT raising when the MNE_USE_CUDA config key is
    not "true" (the default) or cupy is missing, leaving _cuda_capable False.
    Passing n_jobs="cuda" in that state makes MNE coerce n_jobs to 1 before
    falling back to the CPU -- so the run loses its workers and ends up slower
    than never asking for the GPU.
    """
    monkeypatch.setattr(gpu, "_GPU_ENABLED", True)
    monkeypatch.setattr(gpu, "_MNE_CUDA_STATUS", "initialized")

    monkeypatch.setattr(gpu, "_mne_cuda_capable", lambda: False)
    assert gpu.filter_n_jobs(16) == 16

    monkeypatch.setattr(gpu, "_mne_cuda_capable", lambda: True)
    assert gpu.filter_n_jobs(16) == "cuda"


def test_mne_cuda_capable_reads_mne_flag_and_fails_safe(monkeypatch):
    fake_mne = SimpleNamespace(cuda=SimpleNamespace(_cuda_capable=True))
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    assert gpu._mne_cuda_capable() is True

    fake_mne.cuda._cuda_capable = False
    assert gpu._mne_cuda_capable() is False

    # A future MNE that drops the flag must degrade to CPU, not claim CUDA.
    monkeypatch.setitem(sys.modules, "mne", SimpleNamespace(cuda=SimpleNamespace()))
    assert gpu._mne_cuda_capable() is False
