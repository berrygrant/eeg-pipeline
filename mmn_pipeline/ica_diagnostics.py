# mmn_pipeline/ica_diagnostics.py
from __future__ import annotations
import numpy as np
import mne

from .artifacts import moving_window_ptp_mask

def count_clusters(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    m = mask.astype(int)
    return int(np.sum(np.diff(np.r_[0, m]) == 1))

def compute_ica_diagnostics(
    raw: mne.io.BaseRaw,
    *,
    blink_proxy_chs: list[str] | None = None,
    blink_threshold_uv: float = 75.0,
    blink_win_ms: float = 200.0,
    blink_step_ms: float = 10.0,
):
    """
    Compute non-destructive ICA diagnostics on continuous data.
    Returns a dict of metrics.
    """

    sfreq = float(raw.info["sfreq"])

    # ---- Picks ----
    eog_picks = mne.pick_types(raw.info, eog=True, eeg=False)
    eeg_picks = mne.pick_types(raw.info, eog=False, eeg=True)

    metrics = {
        "eog_corr_max": np.nan,
        "eog_corr_mean": np.nan,
        "blink_rate_per_min": np.nan,
        "blink_proxy_rate_per_min": np.nan,
        "blink_source": "none",
    }

    # ---- EOG–EEG correlation ----
    if len(eog_picks) > 0 and len(eeg_picks) > 0:
        eog = raw.get_data(picks=eog_picks)
        eeg = raw.get_data(picks=eeg_picks)

        corr = np.corrcoef(eog, eeg)
        eog_eeg_corr = np.abs(corr[: len(eog_picks), len(eog_picks) :])

        metrics["eog_corr_max"] = float(np.nanmax(eog_eeg_corr))
        metrics["eog_corr_mean"] = float(np.nanmean(eog_eeg_corr))

    # ---- Blink rate per minute ----
    duration_min = raw.times[-1] / 60.0 if raw.times.size else np.nan
    if duration_min and duration_min > 0:

        # Prefer true EOG if present
        if len(eog_picks) > 0:
            blink_mask = moving_window_ptp_mask(
                raw.get_data(picks=eog_picks),
                sfreq=sfreq,
                win_ms=blink_win_ms,
                step_ms=blink_step_ms,
                threshold_uv=blink_threshold_uv,
            )
            blink_events_n = count_clusters(blink_mask)
            metrics["blink_rate_per_min"] = float((blink_events_n) / duration_min)
            metrics["blink_source"] = "eog"

        # Otherwise use frontal EEG proxy channel(s)
        else:
            blink_proxy_chs = blink_proxy_chs or []
            proxy_existing = [ch for ch in blink_proxy_chs if ch in raw.ch_names]

            if proxy_existing:
                proxy_picks = mne.pick_channels(raw.ch_names, include=proxy_existing)
                blink_mask = moving_window_ptp_mask(
                    raw.get_data(picks=proxy_picks),
                    sfreq=sfreq,
                    win_ms=blink_win_ms,
                    step_ms=blink_step_ms,
                    threshold_uv=blink_threshold_uv,
                )
                blink_events_n = count_clusters(blink_mask)
                metrics["blink_proxy_rate_per_min"] = float((blink_events_n) / duration_min)
                metrics["blink_source"] = f"proxy:{','.join(proxy_existing)}"
    return metrics


def recommend_ica(
    *,
    epoch_reject_rate: float,
    eog_corr_max: float,
    blink_rate_per_min: float,
    epoch_loss_thresh: float = 0.20,
    eog_corr_thresh: float = 0.30,
    blink_rate_thresh: float = 20.0,
    blink_proxy_rate_per_min: float,
):
    """
    Decide whether ICA is recommended and explain why.
    """

    reasons = []
    if not np.isnan(blink_rate_per_min) and blink_rate_per_min > blink_rate_thresh:
        reasons.append(f"blink_rate>{blink_rate_thresh:.0f}/min")
    elif not np.isnan(blink_proxy_rate_per_min) and blink_proxy_rate_per_min > blink_rate_thresh:
        reasons.append(f"blink_proxy>{blink_rate_thresh:.0f}/min")

    if epoch_reject_rate > epoch_loss_thresh:
        reasons.append(f"epoch_loss>{epoch_loss_thresh:.2f}")

    if not np.isnan(eog_corr_max) and eog_corr_max > eog_corr_thresh:
        reasons.append(f"eog_corr>{eog_corr_thresh:.2f}")

    if not np.isnan(blink_rate_per_min) and blink_rate_per_min > blink_rate_thresh:
        reasons.append(f"blink_rate>{blink_rate_thresh:.0f}/min")

    return {
        "ica_recommended": bool(reasons),
        "ica_recommend_reason": "+".join(reasons) if reasons else "",
    }