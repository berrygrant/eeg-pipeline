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
