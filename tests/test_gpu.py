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
