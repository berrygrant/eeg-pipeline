# eeg_pipeline/gpu.py
from __future__ import annotations

import numpy as np

_XP = np
_BACKEND = "numpy"
_GPU_ENABLED = False
_MNE_CUDA_STATUS = "disabled"
_CUPY_STATUS = "disabled"
_DEVICE: int | None = None


def _set_backend(xp, name: str, enabled: bool) -> None:
    global _XP, _BACKEND, _GPU_ENABLED
    _XP = xp
    _BACKEND = name
    _GPU_ENABLED = bool(enabled)


def _try_init_mne_cuda(device: int | None) -> str:
    try:
        import mne  # noqa: F401
    except Exception as e:
        return f"mne_unavailable: {e}"

    try:
        cuda_mod = getattr(mne, "cuda", None)
        if cuda_mod is None:
            return "mne_cuda_not_available"

        if device is not None and hasattr(cuda_mod, "set_cuda_device"):
            cuda_mod.set_cuda_device(device)

        if hasattr(cuda_mod, "init_cuda"):
            cuda_mod.init_cuda()
            return "initialized"

        return "available"
    except Exception as e:
        return f"error: {e}"


def _try_init_cupy(device: int | None):
    try:
        import cupy as cp  # type: ignore

        if device is not None:
            cp.cuda.Device(device).use()
        return "available", cp
    except Exception as e:
        return f"unavailable: {e}", None


def configure(use_gpu: bool, device: int | None = None) -> dict:
    """Configure optional GPU acceleration.

    Returns a status dict with backend + availability information.
    """
    global _MNE_CUDA_STATUS, _CUPY_STATUS, _DEVICE
    _DEVICE = device

    if not use_gpu:
        _MNE_CUDA_STATUS = "disabled"
        _CUPY_STATUS = "disabled"
        _set_backend(np, "numpy", False)
        return {
            "requested": False,
            "enabled": False,
            "backend": _BACKEND,
            "mne_cuda": _MNE_CUDA_STATUS,
            "cupy": _CUPY_STATUS,
            "device": _DEVICE,
        }

    _MNE_CUDA_STATUS = _try_init_mne_cuda(device)
    _CUPY_STATUS, xp = _try_init_cupy(device)

    if xp is not None:
        _set_backend(xp, "cupy", True)
        enabled = True
    else:
        # MNE may still use GPU even if CuPy isn't available
        _set_backend(np, "numpy", _MNE_CUDA_STATUS in {"initialized", "available"})
        enabled = _GPU_ENABLED

    return {
        "requested": True,
        "enabled": enabled,
        "backend": _BACKEND,
        "mne_cuda": _MNE_CUDA_STATUS,
        "cupy": _CUPY_STATUS,
        "device": _DEVICE,
    }


def capability_report() -> dict:
    rep = {
        "backend": _BACKEND,
        "gpu_enabled": _GPU_ENABLED,
        "mne_cuda": _MNE_CUDA_STATUS,
        "cupy": _CUPY_STATUS,
        "device": _DEVICE,
    }

    try:
        import mne  # type: ignore

        rep["mne_version"] = getattr(mne, "__version__", "unknown")
    except Exception as e:
        rep["mne_version"] = f"unavailable: {e}"

    try:
        import cupy as cp  # type: ignore

        rep["cupy_version"] = getattr(cp, "__version__", "unknown")
        try:
            rep["gpu_count"] = int(cp.cuda.runtime.getDeviceCount())
        except Exception:
            rep["gpu_count"] = None

        dev = _DEVICE
        if dev is None:
            try:
                dev = int(cp.cuda.runtime.getDevice())
            except Exception:
                dev = None
        rep["device"] = dev

        if dev is not None:
            try:
                props = cp.cuda.runtime.getDeviceProperties(dev)
                name = props.get("name", "") if isinstance(props, dict) else ""
                if isinstance(name, bytes):
                    name = name.decode("utf-8", "replace")
                rep["device_name"] = name

                major = props.get("major", None) if isinstance(props, dict) else None
                minor = props.get("minor", None) if isinstance(props, dict) else None
                if (major is not None) and (minor is not None):
                    rep["compute_capability"] = f"{major}.{minor}"

                total_mem = props.get("totalGlobalMem", None) if isinstance(props, dict) else None
                if total_mem is not None:
                    rep["total_mem_gb"] = round(float(total_mem) / (1024 ** 3), 2)
            except Exception as e:
                rep["device_info_error"] = f"{e}"
    except Exception as e:
        rep["cupy_error"] = f"{e}"

    return rep


def format_capability_report(rep: dict) -> str:
    parts = []
    if rep.get("gpu_count") is not None:
        parts.append(f"gpu_count={rep['gpu_count']}")
    if rep.get("device") is not None:
        parts.append(f"device={rep['device']}")
    if rep.get("device_name"):
        parts.append(f"name={rep['device_name']}")
    if rep.get("compute_capability"):
        parts.append(f"cc={rep['compute_capability']}")
    if rep.get("total_mem_gb") is not None:
        parts.append(f"mem_gb={rep['total_mem_gb']}")
    if rep.get("cupy_version"):
        parts.append(f"cupy={rep['cupy_version']}")
    if rep.get("mne_version"):
        parts.append(f"mne={rep['mne_version']}")
    if rep.get("cupy_error"):
        parts.append(f"cupy_error={rep['cupy_error']}")
    if rep.get("device_info_error"):
        parts.append(f"device_info_error={rep['device_info_error']}")
    if not parts:
        return ""
    return "[GPU] capability: " + ", ".join(parts)


def filter_n_jobs(n_jobs: int = 1):
    """Return the ``n_jobs`` value to pass to MNE FFT filtering/resampling.

    MNE only routes filtering through CUDA when ``n_jobs="cuda"`` — initializing
    CUDA via :func:`configure` does nothing on its own. Without this indirection
    ``use_gpu`` had no effect on filtering at all.

    Falls back to the requested CPU worker count whenever CUDA is unavailable, so
    callers can pass the result straight through. Note that MNE's CUDA support
    covers FFT-based filtering and resampling only: ICA fitting and time-frequency
    decomposition stay on the CPU regardless.
    """
    if _GPU_ENABLED and _MNE_CUDA_STATUS in {"initialized", "available"}:
        return "cuda"
    return n_jobs


def get_xp():
    return _XP


def backend() -> str:
    return _BACKEND


def gpu_enabled() -> bool:
    return _GPU_ENABLED


def to_numpy(x):
    if _BACKEND == "cupy":
        import cupy as cp  # type: ignore

        return cp.asnumpy(x)
    return np.asarray(x)
