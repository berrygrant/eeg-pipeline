# mmn_pipeline/ica_diagnostics.py
from __future__ import annotations
import numpy as np
import mne

from mne.preprocessing import find_eog_events


def count_clusters(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    m = mask.astype(int)
    return int(np.sum(np.diff(np.r_[0, m]) == 1))

def compute_ica_diagnostics(
    raw: mne.io.BaseRaw,
    *,
    modality: str = "eeg",
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
    mode = str(modality).strip().lower()
    if mode not in {"eeg", "meg"}:
        raise ValueError(f"Unsupported modality: {modality!r} (use 'eeg' or 'meg').")

    # ---- Picks ----
    eog_picks = mne.pick_types(raw.info, eog=True, eeg=False)
    if mode == "meg":
        sig_picks = mne.pick_types(raw.info, eog=False, eeg=False, meg=True)
    else:
        sig_picks = mne.pick_types(raw.info, eog=False, eeg=True)

    metrics = {
        "eog_corr_max": np.nan,
        "eog_corr_mean": np.nan,
        "blink_rate_per_min": np.nan,
        "blink_proxy_rate_per_min": np.nan,
        "blink_source": "none",
    }

    # ---- EOG–sensor correlation ----
    if len(eog_picks) > 0 and len(sig_picks) > 0:
        eog = raw.get_data(picks=eog_picks)
        sig = raw.get_data(picks=sig_picks)

        corr = np.corrcoef(eog, sig)
        eog_eeg_corr = np.abs(corr[: len(eog_picks), len(eog_picks) :])

        metrics["eog_corr_max"] = float(np.nanmax(eog_eeg_corr))
        metrics["eog_corr_mean"] = float(np.nanmean(eog_eeg_corr))

    # ---- Blink rate per minute ----
    duration_min = (raw.n_times / sfreq) / 60.0    
    if duration_min and duration_min > 0:

        # Prefer true EOG if present
        if len(eog_picks) > 0:
            eog_name = raw.info["ch_names"][eog_picks[0]]
            try:
                eog_events = find_eog_events(raw, ch_name=eog_name, verbose=False)
                blink_events_n = len(eog_events)
            except Exception:
                blink_events_n = 0  # safe fallback
            metrics["blink_rate_per_min"] = float(blink_events_n / duration_min)
            metrics["blink_source"] = f"eog:{eog_name}"

        # Otherwise use frontal EEG proxy channel(s)
        else:
            blink_proxy_chs = blink_proxy_chs or []
            proxy_existing = [ch for ch in blink_proxy_chs if ch in raw.ch_names]

            if proxy_existing:
                # Use the first proxy as a pseudo-EOG for blink event detection
                proxy_ch = proxy_existing[0]
                raw_tmp = raw.copy()
                raw_tmp.set_channel_types({proxy_ch: "eog"}, on_unit_change="ignore")

                try:
                    eog_events = find_eog_events(raw_tmp, ch_name=proxy_ch, verbose=False)
                    blink_events_n = len(eog_events)
                except Exception:
                    blink_events_n = 0  # safe fallback

                metrics["blink_proxy_rate_per_min"] = float(blink_events_n / duration_min)
                metrics["blink_source"] = f"proxy:{proxy_ch}"
    return metrics


def recommend_ica(
    *,
    epoch_reject_rate: float,
    eog_corr_max: float,
    blink_rate_per_min: float,
    blink_proxy_rate_per_min: float,
    epoch_loss_thresh: float = 0.20,
    eog_corr_thresh: float = 0.30,
    blink_rate_thresh: float = 20.0,
):
    reasons = []

    if np.isfinite(blink_rate_per_min) and blink_rate_per_min > blink_rate_thresh:
        reasons.append(f"blink_rate>{blink_rate_thresh:.0f}/min")
    elif np.isfinite(blink_proxy_rate_per_min) and blink_proxy_rate_per_min > blink_rate_thresh:
        reasons.append(f"blink_proxy>{blink_rate_thresh:.0f}/min")

    if epoch_reject_rate > epoch_loss_thresh:
        reasons.append(f"epoch_loss>{epoch_loss_thresh:.2f}")

    if np.isfinite(eog_corr_max) and eog_corr_max > eog_corr_thresh:
        reasons.append(f"eog_corr>{eog_corr_thresh:.2f}")

    return {
        "ica_recommended": bool(reasons),
        "ica_recommend_reason": "+".join(reasons) if reasons else "",
    }
