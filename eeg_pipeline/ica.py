# mmn_pipeline/ica.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import mne


@dataclass
class ICAParams:
    method: str = "fastica"          # "fastica" | "picard" | "infomax"
    n_components: float | int = 0.99 # variance fraction or int
    random_state: int = 97
    max_iter: int = 512
    fit_l_freq: float = 1.0          # IMPORTANT: fit ICA on >=1 Hz copy
    fit_h_freq: float | None = None  # typically None or 30.0
    corr_thresh: float = 0.30        # for proxy correlation
    max_exclude: int = 3             # max # components to remove
    decim: int = 3                   # speed-up for ICA fit


def _safe_pick_channels(info: mne.Info, names: list[str]) -> list[int]:
    picks = []
    for nm in names:
        if nm in info["ch_names"]:
            picks.append(info["ch_names"].index(nm))
    return picks


def fit_ica(raw: mne.io.BaseRaw, params: ICAParams) -> tuple[mne.preprocessing.ICA, dict[str, Any]]:
    """
    Fit ICA on a COPY of raw filtered to params.fit_l_freq for better stationarity.
    Returns (ica, diag).
    """
    raw_fit = raw.copy()

    # ICA best practice: fit on >=1 Hz high-pass (does not change your analysis filter)
    raw_fit.filter(
        l_freq=params.fit_l_freq,
        h_freq=params.fit_h_freq,
        picks="eeg",
        method="fir",
        phase="zero",
        verbose=False,
    )

    ica = mne.preprocessing.ICA(
        n_components=params.n_components,
        method=params.method,
        random_state=params.random_state,
        max_iter=params.max_iter,
    )

    picks_eeg = mne.pick_types(raw_fit.info, eeg=True, eog=False, ecg=False, stim=False, exclude="bads")
    ica.fit(raw_fit, picks=picks_eeg, decim=params.decim, verbose=False)

    diag = {
        "ica_fit_l_freq": float(params.fit_l_freq),
        "ica_fit_h_freq": None if params.fit_h_freq is None else float(params.fit_h_freq),
        "ica_method": params.method,
        "ica_n_components": int(getattr(ica, "n_components_", len(ica.get_components()))),
        "ica_decim": int(params.decim),
    }
    return ica, diag


def find_ica_excludes(
    ica: mne.preprocessing.ICA,
    raw: mne.io.BaseRaw,
    *,
    eog_chs: list[str] | None = None,
    proxy_chs: list[str] | None = None,
    corr_thresh: float = 0.30,
    max_exclude: int = 3,
) -> tuple[list[int], dict[str, Any]]:
    """
    Identify ocular components.
    Priority:
      1) If EOG channels exist -> use ICA.find_bads_eog
      2) Else -> compute correlation between ICA sources and proxy channel (e.g., Fp1)
    """
    eog_chs = eog_chs or []
    proxy_chs = proxy_chs or ["Fp1"]

    info = raw.info
    eog_picks = _safe_pick_channels(info, eog_chs)
    use_eog = len(eog_picks) > 0

    diag: dict[str, Any] = {
        "ica_blink_source": "eog" if use_eog else "proxy",
        "ica_corr_thresh": float(corr_thresh),
        "ica_max_exclude": int(max_exclude),
    }

    if use_eog:
        # Use the first available EOG channel
        eog_name = info["ch_names"][eog_picks[0]]
        inds, scores = ica.find_bads_eog(raw, ch_name=eog_name, threshold=None, verbose=False)
        # Sort by absolute score descending
        inds_scores = sorted([(i, float(scores[i])) for i in inds], key=lambda x: abs(x[1]), reverse=True)
        keep = [i for i, _ in inds_scores[:max_exclude]]
        diag.update(
            {
                "ica_eog_channel_used": eog_name,
                "ica_candidates": [int(i) for i, _ in inds_scores],
                "ica_scores": [float(s) for _, s in inds_scores],
            }
        )
        return keep, diag

    # Proxy: correlate ICA sources with a frontal EEG channel
    proxy_picks = _safe_pick_channels(info, proxy_chs)
    if len(proxy_picks) == 0:
        diag.update({"ica_proxy_channel_used": "", "ica_error": "no_eog_and_no_proxy_channel_found"})
        return [], diag

    proxy_name = info["ch_names"][proxy_picks[0]]
    diag["ica_proxy_channel_used"] = proxy_name

    # Get ICA sources (components x time)
    sources = ica.get_sources(raw).get_data()  # shape: (n_components, n_times)
    proxy = raw.get_data(picks=[proxy_name]).ravel()

    # Robust correlation (demean, avoid NaNs)
    proxy = proxy - np.nanmean(proxy)
    sources = sources - np.nanmean(sources, axis=1, keepdims=True)

    denom = np.linalg.norm(sources, axis=1) * (np.linalg.norm(proxy) + 1e-12)
    corr = (sources @ proxy) / (denom + 1e-12)
    corr = np.nan_to_num(corr)

    # Pick top components above threshold by abs correlation
    order = np.argsort(-np.abs(corr))
    keep = []
    keep_scores = []
    for idx in order:
        if len(keep) >= max_exclude:
            break
        if abs(corr[idx]) >= corr_thresh:
            keep.append(int(idx))
            keep_scores.append(float(corr[idx]))

    diag.update(
        {
            "ica_candidates": [int(i) for i in order[: min(10, len(order))]],
            "ica_scores": [float(corr[i]) for i in order[: min(10, len(order))]],
            "ica_selected": keep,
            "ica_selected_scores": keep_scores,
        }
    )
    return keep, diag


def apply_ica(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA, exclude: list[int]) -> mne.io.BaseRaw:
    """
    Apply ICA cleaning to raw (in-place on a copy).
    """
    raw_clean = raw.copy()
    ica.exclude = list(map(int, exclude))
    ica.apply(raw_clean, verbose=False)
    return raw_clean