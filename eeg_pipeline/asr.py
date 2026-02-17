# eeg_pipeline/asr.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import mne


@dataclass
class ASRParams:
    enabled: bool = False
    cutoff: float = 20.0
    blocksize: int = 100
    win_len: float = 0.5
    win_overlap: float = 0.66
    max_dropout_fraction: float = 0.1
    min_clean_fraction: float = 0.25
    max_bad_chans: float = 0.1
    method: str = "euclid"
    lookahead: float = 0.25
    stepsize: int = 32
    maxdims: float = 0.66


def apply_asr(raw: mne.io.BaseRaw, params: ASRParams) -> Tuple[mne.io.BaseRaw, Dict[str, Any]]:
    """Apply Artifact Subspace Reconstruction (ASR) if enabled.

    Returns (raw_out, diagnostics). If ASR is disabled or unavailable,
    the input raw is returned unchanged and diagnostics explain why.
    """
    diag: Dict[str, Any] = {
        "asr_enabled": bool(params.enabled),
        "asr_applied": False,
        "asr_cutoff": float(params.cutoff),
        "asr_blocksize": int(params.blocksize),
        "asr_win_len": float(params.win_len),
        "asr_win_overlap": float(params.win_overlap),
        "asr_max_dropout_fraction": float(params.max_dropout_fraction),
        "asr_min_clean_fraction": float(params.min_clean_fraction),
        "asr_max_bad_chans": float(params.max_bad_chans),
        "asr_method": str(params.method),
        "asr_lookahead": float(params.lookahead),
        "asr_stepsize": int(params.stepsize),
        "asr_maxdims": float(params.maxdims),
        "asr_error": "",
    }

    if not params.enabled:
        return raw, diag

    try:
        from asrpy import ASR  # type: ignore
    except Exception as e:  # pragma: no cover - import guard
        diag["asr_error"] = f"asrpy not available: {e}"
        return raw, diag

    try:
        asr = ASR(
            sfreq=float(raw.info["sfreq"]),
            cutoff=float(params.cutoff),
            blocksize=int(params.blocksize),
            win_len=float(params.win_len),
            win_overlap=float(params.win_overlap),
            max_dropout_fraction=float(params.max_dropout_fraction),
            min_clean_fraction=float(params.min_clean_fraction),
            max_bad_chans=float(params.max_bad_chans),
            method=str(params.method),
        )
        asr.fit(raw, picks="eeg")
        raw_out = asr.transform(
            raw,
            picks="eeg",
            lookahead=float(params.lookahead),
            stepsize=int(params.stepsize),
            maxdims=float(params.maxdims),
        )
        diag["asr_applied"] = True
        return raw_out, diag
    except Exception as e:
        diag["asr_error"] = str(e)
        return raw, diag
