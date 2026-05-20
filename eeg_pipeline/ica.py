from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, Union

import numpy as np
import mne
import re


@dataclass
class ICAParams:
    """Container for ICA parameters.

    Parameters
    ----------
    method : str
        The ICA solver to use ("fastica", "picard" or "infomax").
    n_components : float or int
        If float, the fraction of variance to explain (between 0 and 1).  If int,
        the number of principal components to keep.  Defaults to 0.99.
    random_state : int
        Seed for the random number generator used by the ICA solver.
    max_iter : int
        Maximum number of iterations for the ICA solver.
    fit_l_freq : float
        High‑pass frequency applied only before fitting ICA.  Setting this to
        at least 1 Hz improves decomposition stability【639331183304365†L606-L616】.
    fit_h_freq : float or None
        Optional low‑pass frequency applied only before fitting ICA.  None
        leaves the data unfiltered on the upper bound.
    decim : int
        Decimation factor applied during ICA fitting to speed up computation.
    corr_thresh : float
        Absolute correlation threshold used when correlating ICA sources with
        proxy channels to identify ocular components.
    max_exclude : int
        Maximum number of components to mark for exclusion.
    """

    method: str = "fastica"
    n_components: Union[float, int] = 0.99
    random_state: int = 97
    max_iter: int = 512
    fit_l_freq: float = 1.0
    fit_h_freq: Optional[float] = None
    decim: int = 3
    corr_thresh: float = 0.30
    max_exclude: int = 3


def _safe_pick_channels(info: mne.Info, names: list[str]) -> list[int]:
    """Return indices of channels whose names exist in the info object.

    Parameters
    ----------
    info : mne.Info
        The MNE info object containing channel names.
    names : list of str
        List of channel names to pick.

    Returns
    -------
    list of int
        Indices into ``info['ch_names']`` for the channels that were found.  If
        none of the requested channels exist, an empty list is returned.
    """
    picks: list[int] = []
    for nm in names:
        if nm in info["ch_names"]:
            picks.append(info["ch_names"].index(nm))
    return picks


def fit_ica(raw: mne.io.BaseRaw, params: ICAParams) -> Tuple[Optional[mne.preprocessing.ICA], Dict[str, Any]]:
    """
    Fit ICA on a filtered copy of `raw` (ICA-only filtering), returning (ica, diagnostics).
    If ICA cannot be fit (e.g., PCA collapses to 1 component), returns (None, diag) instead of crashing.
    """
    diag: Dict[str, Any] = {
        "ica_fit_ok": False,
        "ica_fit_error": "",
        "ica_fit_retry": "",
        "ica_fit_n_components_used": None,
        "ica_fit_n_eeg_chs": None,
    }

    # --- Make the data we fit ICA on (do NOT mutate the main raw) ---
    raw_fit = raw.copy()

    # ICA is usually fit with a 1 Hz high-pass to stabilize decomposition
    raw_fit.filter(
        l_freq=params.fit_l_freq,
        h_freq=params.fit_h_freq,
        picks="eeg",
        method="fir",
        phase="zero",
        verbose=False,
    )

    # --- EEG picks ---
    picks_eeg = mne.pick_types(raw_fit.info, eeg=True, eog=False, meg=False, stim=False, misc=False)
    diag["ica_fit_n_eeg_chs"] = int(len(picks_eeg))

    if len(picks_eeg) < 2:
        diag["ica_fit_error"] = f"Need >=2 EEG channels for ICA, got {len(picks_eeg)}."
        return None, diag

    def _try_fit(n_components: Union[float, int]) -> mne.preprocessing.ICA:
        ica = mne.preprocessing.ICA(
            method=params.method,
            n_components=n_components,
            random_state=params.random_state,
            max_iter=params.max_iter,
        )
        ica.fit(raw_fit, picks=picks_eeg, decim=params.decim, verbose=False)
        return ica

    # --- First attempt (whatever user/config requested) ---
    try:
        ica = _try_fit(params.n_components)
        diag["ica_fit_ok"] = True
        diag["ica_fit_n_components_used"] = params.n_components
        return ica, diag

    except RuntimeError as e:
        msg = str(e)
        diag["ica_fit_error"] = msg

        # MNE variance-fraction edge case -> PCA collapses to 1 component
        if "your threshold results in 1 component" in msg:
            # Retry with an int number of components
            # Use min(max(15, n_ch-1), 40) as a safe default
            n_ch = len(picks_eeg)
            fallback_n = int(min(max(15, n_ch - 1), 40))
            diag["ica_fit_retry"] = f"retry_int_n_components={fallback_n}"

            try:
                ica = _try_fit(fallback_n)
                diag["ica_fit_ok"] = True
                diag["ica_fit_n_components_used"] = fallback_n
                return ica, diag
            except Exception as e2:
                diag["ica_fit_error"] = f"{msg} | retry failed: {e2}"
                return None, diag

        # Other ICA failures should still just skip ICA (do not crash pipeline)
        return None, diag


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
    Identify ocular components and return a list of components to exclude along with diagnostics.

    The search proceeds in two stages:
      1. If any EOG channels are present, use :meth:`ICA.find_bads_eog` to detect EOG-related
         components.  Components are sorted by the absolute value of their correlation scores
         and the top ``max_exclude`` are returned.  This leverages MNE's adaptive z‑scoring
         which determines an appropriate correlation threshold automatically【52308007362329†L1406-L1465】.
      2. If no EOG channels exist, fall back to correlating ICA sources with a proxy frontal EEG
         channel (e.g., ``Fp1``).  Components whose absolute correlation with the proxy exceeds
         ``corr_thresh`` are considered ocular; the top ``max_exclude`` components are returned.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        The fitted ICA object.
    raw : mne.io.BaseRaw
        The raw data used for computing ICA sources.
    eog_chs : list of str | None
        Names of EOG channels to use for blink detection.  If provided and at least one
        exists in ``raw``, the EOG-based detection is used.
    proxy_chs : list of str | None
        Names of EEG channels to use as blink proxies if no EOG channels exist.  Defaults
        to ["Fp1"].
    corr_thresh : float
        Absolute correlation threshold when using proxy channels.  Only components whose
        absolute correlation exceeds this value are considered ocular.
    max_exclude : int
        Maximum number of components to mark for exclusion.

    Returns
    -------
    exclude : list of int
        Indices of ICA components to exclude.
    diag : dict
        Diagnostic information about the selection, including which channel was used and
        correlation scores.
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

    # Case 1: true EOG channels exist -> use adaptive z‑scoring from MNE
    if use_eog:
        # Use the first available EOG channel for blink detection
        eog_name = info["ch_names"][eog_picks[0]]
        inds, scores = ica.find_bads_eog(raw, ch_name=eog_name, threshold="auto", verbose=False)
        # Sort identified components by absolute score (descending)
        inds_scores = sorted(
            [(i, float(scores[i])) for i in inds], key=lambda x: abs(x[1]), reverse=True
        )
        # Take up to max_exclude components
        exclude = [int(i) for i, _ in inds_scores[:max_exclude]]
        diag.update(
            {
                "ica_eog_channel_used": eog_name,
                "ica_candidates": [int(i) for i, _ in inds_scores],
                "ica_scores": [float(s) for _, s in inds_scores],
                "ica_excluded": exclude,
            }
        )
        return exclude, diag

    # Case 2: no EOG -> correlate ICA sources with a proxy EEG channel
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

    # Compute Pearson correlations for each component and proxy
    denom = np.linalg.norm(sources, axis=1) * (np.linalg.norm(proxy) + 1e-12)
    corr = (sources @ proxy) / (denom + 1e-12)
    corr = np.nan_to_num(corr)

    # Rank components by absolute correlation
    order = np.argsort(-np.abs(corr))
    exclude: list[int] = []
    exclude_scores: list[float] = []
    for idx in order:
        if len(exclude) >= max_exclude:
            break
        if abs(corr[idx]) >= corr_thresh:
            exclude.append(int(idx))
            exclude_scores.append(float(corr[idx]))

    diag.update(
        {
            "ica_candidates": [int(i) for i in order[: min(10, len(order))]],
            "ica_scores": [float(corr[i]) for i in order[: min(10, len(order))]],
            "ica_excluded": exclude,
            "ica_excluded_scores": exclude_scores,
        }
    )
    return exclude, diag


def apply_ica(raw: mne.io.BaseRaw, ica: mne.preprocessing.ICA, exclude: list[int]) -> mne.io.BaseRaw:
    """
    Apply ICA cleaning to raw (in-place on a copy).
    """
    raw_clean = raw.copy()
    ica.exclude = list(map(int, exclude))
    ica.apply(raw_clean, verbose=False)
    return raw_clean
