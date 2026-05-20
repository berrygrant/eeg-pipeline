from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from .align import (
    align_marker_positions_to_codes,
    detect_trigger_bursts,
    format_alignment_diag,
)
from .artifacts import (
    moving_window_ptp_mask,
    moving_window_ptp_max,
    simple_voltage_threshold_mask,
    step_threshold_mask,
)
from .behavior import (
    clean_eventcodes,
    filter_codes,
    read_eventcodes_from_subject_csv,
    resolve_subject_csv_path,
    subject_number_from_stem,
)
from .epoching import (
    EpochParams,
    build_events_from_positions_and_codes,
    make_epochs,
    select_and_filter_conditions,
    select_and_recode_stddev,
)
from .evoked import compute_evokeds, grand_averages
from .ica import ICAParams, apply_ica, find_ica_excludes, fit_ica
from .ica_diagnostics import compute_ica_diagnostics, recommend_ica
from .io_brainvision import (
    brainvision_links_ok,
    events_from_annotations_positions,
    read_raw_preprocess,
)
from .metrics.writers import ParquetRowGroupWriter, reset_combined_metric_outputs
from .metrics_runner import resolve_metrics_conditions, write_subject_metrics
from .pipeline_config import apply_config
from .qc import write_qc_summary
from .schema import derive_metadata_from_condition_map, derive_metadata_v1, parse_token_map


def _parse_n_components(x):
    """
    MNE ICA n_components can be float (variance fraction) or int (#components).
    argparse gives us a string; infer int vs float.
    """
    if x is None:
        return 0.99
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip()
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return float(s)


def run_full_pipeline(args, defaults=None, cfg=None):
    """Run the full EEG processing pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    defaults : dict or None
        Mapping from argument names to their argparse defaults.  Used to
        determine whether CLI arguments were explicitly provided.  If None,
        an empty dict is used and all config values will be applied.
    """
    if defaults is None:
        defaults = {}
    if cfg is None:
        cfg = apply_config(args, defaults)

    raw_dir = Path(args.raw_dir)
    subject_csv_dir = Path(args.subject_csv_dir)
    out_dir = Path(args.out_dir)
    prepare_output_dirs(out_dir)

    d_raw = out_dir / "01_clean_raw"
    d_epo = out_dir / "02_epochs"
    d_evk = out_dir / "03_evokeds"
    d_ga = out_dir / "04_grand_averages"
    metrics_dir = out_dir / "05_metrics"
    for d in (d_raw, d_epo, d_evk, d_ga):
        d.mkdir(parents=True, exist_ok=True)
    if int(getattr(args, "metrics", 0)):
        metrics_dir.mkdir(parents=True, exist_ok=True)
        reset_combined_metric_outputs(metrics_dir)

    ep = EpochParams(
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=(float(args.baseline[0]), float(args.baseline[1])),
    )

    token_map = parse_token_map(args.token_map)

    rows: list[dict] = []
    evokeds_by_cond: dict[str, list[mne.Evoked]] = {}

    parquet_writer = ParquetRowGroupWriter()

    raw_files = [p for p in raw_dir.rglob("*.vhdr") if p.is_file() and ".git" not in p.parts]
    raw_files = sorted(raw_files)
    if not raw_files:
        raw_files = [p for p in raw_dir.rglob("*.set") if p.is_file() and ".git" not in p.parts]
        raw_files = sorted(raw_files)
    if not raw_files:
        raise RuntimeError(f"No .vhdr or .set files found in {raw_dir}")

    if args.subjects:
        wanted = {s.lower() for s in args.subjects}
        raw_files = [p for p in raw_files if p.stem.lower() in wanted]
        if not raw_files:
            raise RuntimeError(f"No matching raw files found for --subjects={args.subjects}")

    std_codes = np.asarray(args.standard_codes, dtype=int)
    dev_codes = np.asarray(args.deviant_codes, dtype=int)
    stddev_set = np.r_[std_codes, dev_codes]

    condition_map = getattr(args, "condition_map", None)
    metrics_conditions = resolve_metrics_conditions(args)

    for raw_path in raw_files:
        subj = raw_path.stem
        subj_num = subject_number_from_stem(subj)
        subject_csv = resolve_subject_csv_path(subject_csv_dir, subj_num, subj)
        subject_csv_name = subject_csv.name
        subject_csv_path = str(subject_csv)
        subject_csv_exists = bool(subject_csv.exists())
        is_bv = raw_path.suffix.lower() == ".vhdr"
        vmrk = raw_path.with_suffix(".vmrk") if is_bv else None

        print(f"\n=== {subj} ===")

        # Always define burst QC fields so the CSV schema is consistent
        burst_qc = {
            "trigger_burst_flag": False,
            "trigger_n_short_iti": 0,
            "trigger_min_iti_s": "",
            "trigger_burst_max_in_window": "",
            "trigger_burst_n_windows_ge_thresh": 0,
            "trigger_burst_params": "",
        }

        if is_bv:
            if not vmrk or not vmrk.exists():
                msg = f"Missing .vmrk for {subj}: {vmrk}"
                if args.on_missing_vmrk == "fail":
                    raise FileNotFoundError(msg)
                if args.on_missing_vmrk == "skip":
                    print("[WARN]", msg, "-> skipping")
                    rows.append(
                        {
                            "subject": subj,
                            "raw_file": str(raw_path.name),
                            "subject_csv": subject_csv_name,
                            "subject_csv_path": subject_csv_path,
                            "subject_csv_exists": subject_csv_exists,
                            **burst_qc,
                            "status": "SKIP_MISSING_VMRK",
                            "error": msg,
                        }
                    )
                    continue
                print("[WARN]", msg)

            ok, reason = brainvision_links_ok(raw_path)
            if not ok:
                msg = f"BrainVision link mismatch in {raw_path.name}: {reason}"
                if args.on_bv_link_mismatch == "fail":
                    raise FileNotFoundError(msg)
                print("[WARN]", msg, "-> skipping")
                rows.append(
                        {
                            "subject": subj,
                            "raw_file": str(raw_path.name),
                            "subject_csv": subject_csv_name,
                            "subject_csv_path": subject_csv_path,
                            "subject_csv_exists": subject_csv_exists,
                            **burst_qc,
                            "status": "SKIP_BV_LINK_MISMATCH",
                            "error": msg,
                        }
                )
                continue

        raw = read_raw_preprocess(
            raw_path=raw_path,
            montage=args.montage,
            eog_chs=args.eog_chs,
            aux_chs=args.aux_chs,
            reref=args.reref,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            notch=args.notch,
        )

        ica_diag = compute_ica_diagnostics(
            raw,
            blink_proxy_chs=args.blink_proxy_chs,
            blink_threshold_uv=args.blink_threshold_uv,
            blink_win_ms=args.blink_win_ms,
            blink_step_ms=args.blink_step_ms,
        )

        # ---- ICA: optional fit + apply (before event extraction / epoching) ----
        ica_ran = False
        ica_applied = False
        ica_exclude: list[int] = []
        ica_fit_diag: dict = {}
        ica_find_diag: dict = {}

        if args.ica == "auto":
            do_ica = False

            rate = float(ica_diag.get("blink_rate_per_min", np.nan))
            proxy_rate = float(ica_diag.get("blink_proxy_rate_per_min", np.nan))
            blink_rate = rate if np.isfinite(rate) and rate > 0 else proxy_rate
            max_corr = float(ica_diag.get("eog_corr_max", np.nan))

            if np.isfinite(blink_rate) and blink_rate >= args.ica_auto_blink_rate_per_min:
                do_ica = True
            elif np.isfinite(max_corr) and max_corr >= args.ica_corr_thresh:
                do_ica = True

            if do_ica:
                ica_params = ICAParams(
                    method=args.ica_method,
                    n_components=_parse_n_components(args.ica_n_components),
                    random_state=args.ica_random_state,
                    max_iter=args.ica_max_iter,
                    fit_l_freq=args.ica_fit_l_freq,
                    fit_h_freq=args.ica_fit_h_freq,
                    corr_thresh=args.ica_corr_thresh,
                    max_exclude=args.ica_max_exclude,
                    decim=args.ica_decim,
                )

                ica_obj, ica_fit_diag = fit_ica(raw, ica_params)
                if ica_obj is None:
                    print(f"[WARN] ICA fit failed for {subj}; continuing without ICA.")
                else:
                    ica_ran = True
                    ica_exclude, ica_find_diag = find_ica_excludes(
                        ica_obj,
                        raw,
                        eog_chs=args.eog_chs,
                        proxy_chs=args.blink_proxy_chs,
                        corr_thresh=args.ica_corr_thresh,
                        max_exclude=args.ica_max_exclude,
                    )
                    if len(ica_exclude) > 0:
                        raw = apply_ica(raw, ica_obj, ica_exclude)
                        ica_applied = True
                    if bool(args.save_ica):
                        ica_path = out_dir / "00_ica" / f"{subj}-ica.fif"
                        ica_path.parent.mkdir(parents=True, exist_ok=True)
                        ica_obj.save(ica_path, overwrite=True)

        # Events from annotations
        events_ann = events_from_annotations_positions(raw)
        markers_pos = events_ann[:, 0].copy()
        
        # Trigger burst QC (flag only; do not modify markers_pos)
        burst_diag = detect_trigger_bursts(
            markers_pos=markers_pos,
            sfreq=float(raw.info["sfreq"]),
            min_iti_s=0.02,
            burst_win_s=0.25,
            burst_count=5,
        )
        burst_qc = {
            "trigger_burst_flag": bool(burst_diag.get("burst_flag", False)),
            "trigger_n_short_iti": int(burst_diag.get("n_short_iti", 0) or 0),
            "trigger_min_iti_s": burst_diag.get("min_iti_s", ""),
            "trigger_burst_max_in_window": int(burst_diag.get("burst_max_in_window", 1) or 1),
            "trigger_burst_n_windows_ge_thresh": int(burst_diag.get("burst_n_windows_ge_thresh", 0) or 0),
            "trigger_burst_params": burst_diag.get("burst_params", ""),
        }
        if burst_qc["trigger_burst_flag"]:
            print(
                f"[WARN] Trigger burst detected for {subj}: "
                f"short_iti={burst_qc['trigger_n_short_iti']}, "
                f"max_in_window={burst_qc['trigger_burst_max_in_window']}"
            )

        if not subject_csv.exists():
            msg = f"Missing subject file for {subj}: {subject_csv}"
            if args.on_missing_subject_csv == "fail":
                raise FileNotFoundError(msg)
            print("[WARN]", msg, "-> skipping")
            rows.append(
                    {
                        "subject": subj,
                        "raw_file": str(raw_path.name),
                        "subject_csv": subject_csv_name,
                        "subject_csv_path": subject_csv_path,
                        "subject_csv_exists": subject_csv_exists,
                        **burst_qc,
                        "status": "SKIP_MISSING_SUBJECT_CSV",
                        "error": msg,
                    }
            )
            continue

        try:
            codes_raw = read_eventcodes_from_subject_csv(subject_csv)
            codes_all, cleanup_diag = clean_eventcodes(codes_raw, args.eventcode_cleanup)
        except Exception as e:
            msg = f"Behavioral EventCode cleanup failed for {subj}: {e}"
            print("[WARN]", msg, "-> skipping")
            rows.append(
                    {
                        "subject": subj,
                        "raw_file": str(raw_path.name),
                        "subject_csv": subject_csv_name,
                        "subject_csv_path": subject_csv_path,
                        "subject_csv_exists": subject_csv_exists,
                        **burst_qc,
                        "status": "SKIP_EVENTCODE_CLEANUP_FAILED",
                        "error": msg,
                    }
            )
            continue
        if cleanup_diag["eventcode_cleanup_removed"] > 0:
            print(
                f"[INFO] EventCode cleanup {cleanup_diag['eventcode_cleanup_mode']} removed "
                f"{cleanup_diag['eventcode_cleanup_removed']} rows across "
                f"{cleanup_diag['eventcode_cleanup_runs']} runs for {subj}."
            )
        codes = filter_codes(codes_all, args.behavioral_keep_codes)
        try:
            markers_aligned, diag = align_marker_positions_to_codes(
                markers_pos=markers_pos,
                sfreq=float(raw.info["sfreq"]),
                codes=codes,
                gap_s=args.drop_eeg_markers_by_gap_s,
                auto_drop_to_count=bool(args.auto_drop_to_count),
                collapse_bursts_s=args.collapse_eeg_marker_bursts_s,
                collapse_keep=args.collapse_eeg_marker_bursts_keep,
            )
        except Exception as e:
            msg = f"Alignment failed for {subj}: {e}"
            print("[WARN]", msg, "-> skipping")
            rows.append(
                    {
                        "subject": subj,
                        "raw_file": str(raw_path.name),
                        "subject_csv": subject_csv_name,
                        "subject_csv_path": subject_csv_path,
                        "subject_csv_exists": subject_csv_exists,
                        **burst_qc,
                        "status": "SKIP_ALIGNMENT_FAILED",
                        "error": msg,
                    }
            )
            continue

        review_flag = False
        review_reasons = []

        # 1) StimTrak burst
        if burst_qc["trigger_burst_flag"]:
            review_flag = True
            review_reasons.append("trigger_burst")

        # 2) Huge marker excess before alignment (StimTrak spam)
        if diag.get("markers_original", 0) > 2 * len(codes):
            review_flag = True
            review_reasons.append("markers>>behavior")

        # 3) Big auto-drop suggests trigger noise
        if diag.get("markers_dropped_by_auto", 0) >= 50:
            review_flag = True
            review_reasons.append("large_auto_drop")

        # 4) Too few markers vs expected (typically a recording/annotation problem)
        if diag.get("markers_original", 0) < 0.9 * len(codes):
            review_flag = True
            review_reasons.append("markers<behavior")

        events = build_events_from_positions_and_codes(markers_aligned, codes)
        events_stddev, event_id = select_and_recode_stddev(events, args.standard_codes, args.deviant_codes)
        if len(events_stddev) == 0:
            msg = f"No standard/deviant events after filtering for {subj}"
            print("[WARN]", msg, "-> skipping")
            rows.append(
                {
                    "subject": subj,
                    "raw_file": str(raw_path.name),
                    "subject_csv": subject_csv_name,
                    "subject_csv_path": subject_csv_path,
                    "subject_csv_exists": subject_csv_exists,
                    **burst_qc,
                    "status": "SKIP_NO_STDDEV_EVENTS",
                    "error": msg,
                }
            )
            continue
        if condition_map:
            events_epo, event_id, cond_codes = select_and_filter_conditions(events, condition_map)
            keep_mask = np.isin(events[:, 2], np.asarray(cond_codes, dtype=int))
            md_full = derive_metadata_from_condition_map(codes.tolist(), condition_map)
        else:
            events_epo, event_id = select_and_recode_stddev(events, args.standard_codes, args.deviant_codes)
            keep_mask = np.isin(events[:, 2], stddev_set)
            md_full = derive_metadata_v1(codes.tolist(), token_map=token_map)

        if len(events_epo) == 0:
            reason = "condition_map" if condition_map else "standard/deviant codes"
            msg = f"No matching events after applying {reason}; skipping."
            print("[WARN]", msg)
            rows.append(
                {
                    "subject": subj,
                    "raw_file": str(raw_path.name),
                    "subject_csv": subject_csv_name,
                    "subject_csv_path": subject_csv_path,
                    "subject_csv_exists": subject_csv_exists,
                    **diag,
                    **burst_qc,
                    "n_events_used": int(len(events)),
                    "n_events_kept_stddev": 0,
                    "status": "SKIP_NO_CONDITION_EVENTS",
                    "error": msg,
                }
            )
            raw.save(d_raw / f"{subj}-raw.fif", overwrite=True)
            continue

        epochs = make_epochs(raw, events_epo, event_id, ep)
        md = md_full.loc[keep_mask].reset_index(drop=True)
        # Align metadata with epochs that survive MNE's internal dropping
        if len(md) != len(epochs):
            md = md.iloc[epochs.selection].reset_index(drop=True)
        epochs.metadata = md

        epochs_test = epochs.copy().crop(tmin=args.art_test_tmin, tmax=args.art_test_tmax)

        eog_picks = mne.pick_types(epochs_test.info, eog=True, eeg=False)
        blink_threshold_uv = float(args.blink_threshold_uv)
        blink_auto_pct = getattr(args, "blink_auto_percentile", None)
        if blink_auto_pct in ("None", "null"):
            blink_auto_pct = None
        if blink_auto_pct is not None:
            blink_auto_pct = float(blink_auto_pct)
        blink_picks = eog_picks
        if len(blink_picks) == 0:
            proxy = [ch for ch in args.blink_proxy_chs if ch in epochs_test.ch_names]
            if proxy:
                blink_picks = mne.pick_channels(epochs_test.ch_names, include=proxy)
        blink_data = (
            epochs_test.get_data(picks=blink_picks) if len(blink_picks) > 0 else np.empty((0,))
        )
        if blink_auto_pct is not None and len(blink_picks) > 0:
            if blink_data.size:
                ptp_max = moving_window_ptp_max(
                    blink_data,
                    sfreq=float(epochs_test.info["sfreq"]),
                    win_ms=args.blink_win_ms,
                    step_ms=args.blink_step_ms,
                )
                if np.isfinite(ptp_max).any():
                    blink_threshold_uv = float(np.nanpercentile(ptp_max, blink_auto_pct))
        blink_bad = np.zeros(len(epochs_test), dtype=bool)

        if len(blink_picks) > 0:
            blink_bad = moving_window_ptp_mask(
                blink_data,
                sfreq=float(epochs_test.info["sfreq"]),
                win_ms=args.blink_win_ms,
                step_ms=args.blink_step_ms,
                threshold_uv=blink_threshold_uv,
            )

        eeg_picks = mne.pick_types(epochs_test.info, eeg=True, eog=False)
        volt_method = str(getattr(args, "volt_method", "simple")).lower()
        volt_pos_uv = float(args.volt_pos_uv)
        volt_neg_uv = float(args.volt_neg_uv)
        volt_threshold_uv = float(getattr(args, "volt_threshold_uv", 150.0))
        volt_auto_pct = getattr(args, "volt_auto_percentile", None)
        if volt_auto_pct in ("None", "null"):
            volt_auto_pct = None
        if volt_auto_pct is not None:
            volt_auto_pct = float(volt_auto_pct)
        eeg_data = (
            epochs_test.get_data(picks=eeg_picks) if len(eeg_picks) > 0 else np.empty((0,))
        )
        if volt_auto_pct is not None and len(eeg_picks) > 0:
            if eeg_data.size:
                max_abs = np.nanmax(np.abs(eeg_data) * 1e6, axis=(1, 2))
                if np.isfinite(max_abs).any():
                    thr_abs = float(np.nanpercentile(max_abs, volt_auto_pct))
                    if volt_method in {"simple", "combined"}:
                        volt_pos_uv = thr_abs
                        volt_neg_uv = -thr_abs
                if volt_method in {"window_ptp", "combined"}:
                    ptp_max = moving_window_ptp_max(
                        eeg_data,
                        sfreq=float(epochs_test.info["sfreq"]),
                        win_ms=float(getattr(args, "volt_win_ms", 200.0)),
                        step_ms=float(getattr(args, "volt_step_ms", 10.0)),
                    )
                    if np.isfinite(ptp_max).any():
                        volt_threshold_uv = float(np.nanpercentile(ptp_max, volt_auto_pct))
        if len(eeg_picks) == 0:
            muscle_bad = np.zeros(len(epochs_test), dtype=bool)
        elif volt_method == "window_ptp":
            muscle_bad = moving_window_ptp_mask(
                eeg_data,
                sfreq=float(epochs_test.info["sfreq"]),
                win_ms=float(getattr(args, "volt_win_ms", 200.0)),
                step_ms=float(getattr(args, "volt_step_ms", 10.0)),
                threshold_uv=volt_threshold_uv,
            )
        elif volt_method == "combined":
            simple_bad = simple_voltage_threshold_mask(
                eeg_data,
                pos_limit_uv=volt_pos_uv,
                neg_limit_uv=volt_neg_uv,
            )
            ptp_bad = moving_window_ptp_mask(
                eeg_data,
                sfreq=float(epochs_test.info["sfreq"]),
                win_ms=float(getattr(args, "volt_win_ms", 200.0)),
                step_ms=float(getattr(args, "volt_step_ms", 10.0)),
                threshold_uv=volt_threshold_uv,
            )
            muscle_bad = simple_bad | ptp_bad
        else:
            muscle_bad = simple_voltage_threshold_mask(
                eeg_data,
                pos_limit_uv=volt_pos_uv,
                neg_limit_uv=volt_neg_uv,
            )

        step_thresh = getattr(args, "volt_step_uv_per_ms", None)
        if step_thresh not in (None, "None", "null") and len(eeg_picks) > 0:
            step_bad = step_threshold_mask(
                eeg_data,
                sfreq=float(epochs_test.info["sfreq"]),
                threshold_uv_per_ms=float(step_thresh),
            )
            muscle_bad = muscle_bad | step_bad

        threshold_info = {
            "blink_threshold_uv_used": float(blink_threshold_uv),
            "blink_auto_percentile": "" if blink_auto_pct is None else float(blink_auto_pct),
            "volt_pos_uv_used": float(volt_pos_uv),
            "volt_neg_uv_used": float(volt_neg_uv),
            "volt_ptp_threshold_uv_used": (
                float(volt_threshold_uv) if volt_method in {"window_ptp", "combined"} else ""
            ),
            "volt_auto_percentile": "" if volt_auto_pct is None else float(volt_auto_pct),
            "volt_method": volt_method,
        }

        bad = blink_bad | muscle_bad
        bad_idx = np.where(bad)[0].tolist()

        n_before = len(epochs)
        if bad_idx:
            epochs.drop(bad_idx, reason="ARTIFACT_REJECT_MNE")
        n_after = len(epochs)

        if n_after == 0:
            msg = "All epochs dropped after artifact rejection; skipping evoked computation."
            print("[WARN]", msg)
            rows.append(
                    {
                        "subject": subj,
                        "raw_file": str(raw_path.name),
                        "subject_csv": subject_csv_name,
                        "subject_csv_path": subject_csv_path,
                        "subject_csv_exists": subject_csv_exists,
                        **diag,
                        **burst_qc,
                        **threshold_info,
                        "n_epochs_before_artifact": int(n_before),
                        "n_epochs_final": 0,
                    "status": "SKIP_EMPTY_EPOCHS",
                    "error": msg,
                }
            )
            raw.save(d_raw / f"{subj}-raw.fif", overwrite=True)
            continue

        if condition_map:
            n_std = int(np.isin(epochs.events[:, 2], std_codes).sum())
            n_dev = int(np.isin(epochs.events[:, 2], dev_codes).sum())
        else:
            n_std = len(epochs["Standard"])
            n_dev = len(epochs["Deviant"])

        if (not condition_map) and (n_std == 0 or n_dev == 0):
            msg = f"Empty condition after rejection (Standard={n_std}, Deviant={n_dev}); skipping evokeds."
            print("[WARN]", msg)
            rows.append(
                    {
                        "subject": subj,
                        "raw_file": str(raw_path.name),
                        "subject_csv": subject_csv_name,
                        "subject_csv_path": subject_csv_path,
                        "subject_csv_exists": subject_csv_exists,
                        **diag,
                        **burst_qc,
                        **threshold_info,
                        "n_epochs_before_artifact": int(n_before),
                        "n_epochs_final": int(n_after),
                    "n_standard_final": int(n_std),
                    "n_deviant_final": int(n_dev),
                    "status": "SKIP_EMPTY_CONDITION",
                    "error": msg,
                }
            )
            raw.save(d_raw / f"{subj}-raw.fif", overwrite=True)
            epochs.save(d_epo / f"{subj}-epo.fif", overwrite=True)
            continue

        epoch_reject_rate = (n_before - n_after) / n_before if n_before > 0 else 0.0
        max_rr = getattr(args, "max_reject_rate", None)
        if max_rr is not None and epoch_reject_rate > float(max_rr):
            msg = (
                f"Epoch reject rate {epoch_reject_rate:.3f} exceeds max_reject_rate={float(max_rr):.3f}; "
                "excluding subject from evoked/metrics."
            )
            print("[WARN]", msg)
            rows.append(
                {
                    "subject": subj,
                    "raw_file": str(raw_path.name),
                    "subject_csv": subject_csv_name,
                    "subject_csv_path": subject_csv_path,
                    "subject_csv_exists": subject_csv_exists,
                    **diag,
                    **burst_qc,
                    **threshold_info,
                    "n_epochs_before_artifact": int(n_before),
                    "n_epochs_final": int(n_after),
                    "n_standard_final": int(n_std),
                    "n_deviant_final": int(n_dev),
                    "epoch_reject_rate": float(epoch_reject_rate),
                    "status": "SKIP_REJECT_RATE",
                    "error": msg,
                }
            )
            raw.save(d_raw / f"{subj}-raw.fif", overwrite=True)
            epochs.save(d_epo / f"{subj}-epo.fif", overwrite=True)
            continue

        # ------------------------------------------------------------------
        # Metrics (ERP + TFR)
        # ------------------------------------------------------------------
        if int(getattr(args, "metrics", 0)):
            do_erp = bool(getattr(args, "metrics_erp_enabled", True))
            do_tfr = bool(getattr(args, "metrics_tfr_enabled", True))

            if do_erp or do_tfr or bool(getattr(args, "metrics_erp_timeseries", False)):
                write_subject_metrics(
                    epochs=epochs,
                    subject=subj,
                    args=args,
                    metrics_dir=metrics_dir,
                    conditions=metrics_conditions,
                    parquet_writer=parquet_writer,
                )

        ica_recommendation = recommend_ica(
            epoch_reject_rate=epoch_reject_rate,
            eog_corr_max=ica_diag.get("eog_corr_max", 0.0),
            blink_rate_per_min=ica_diag.get("blink_rate_per_min", 0.0),
            blink_proxy_rate_per_min=ica_diag.get("blink_proxy_rate_per_min", 0.0),
            epoch_loss_thresh=0.20,
            eog_corr_thresh=args.ica_corr_thresh,
            blink_rate_thresh=args.ica_auto_blink_rate_per_min,
        )

        raw.save(d_raw / f"{subj}-raw.fif", overwrite=True)
        epochs.save(d_epo / f"{subj}-epo.fif", overwrite=True)

        evoked_conditions = list(event_id.keys())
        evokeds = compute_evokeds(epochs, evoked_conditions)
        for cond, ev in evokeds.items():
            ev.save(d_evk / f"{subj}_{cond}-ave.fif", overwrite=True)
            evokeds_by_cond.setdefault(cond, []).append(ev)

        rows.append(
            {
                "subject": subj,
                "raw_file": str(raw_path.name),
                "subject_csv": subject_csv_name,
                        "subject_csv_path": subject_csv_path,
                        "subject_csv_exists": subject_csv_exists,
                "sfreq": float(raw.info["sfreq"]),
                "token1": token_map.get("token1"),
                "token2": token_map.get("token2"),
                "behavioral_codes_total": int(len(codes_all)),
                "behavioral_codes_used": int(len(codes)),
                "behavioral_keep_codes": " ".join(map(str, args.behavioral_keep_codes)) if args.behavioral_keep_codes else "",
                **diag,
                **burst_qc,
                **threshold_info,
                "n_events_used": int(len(events)),
                "n_events_kept_stddev": int(len(events_epo)),
                "n_epochs_before_artifact": int(n_before),
                "n_blink_bad": int(blink_bad.sum()),
                "n_muscle_bad": int(muscle_bad.sum()),
                "n_epochs_dropped": int(n_before - n_after),
                "n_epochs_final": int(n_after),
                "n_standard_final": int(n_std),
                "n_deviant_final": int(n_dev),
                "epoch_reject_rate": float(epoch_reject_rate),
                "eog_corr_max": float(ica_diag.get("eog_corr_max", 0.0) or 0.0),
                "eog_corr_mean": float(ica_diag.get("eog_corr_mean", 0.0) or 0.0),
                "blink_rate_per_min": float(ica_diag.get("blink_rate_per_min", 0.0) or 0.0),
                "blink_proxy_rate_per_min": float(ica_diag.get("blink_proxy_rate_per_min", 0.0) or 0.0),
                "blink_source": ica_diag.get("blink_source", ""),
                "ica_recommended": bool(ica_recommendation.get("ica_recommended", False)),
                "ica_recommend_reason": ica_recommendation.get("ica_recommend_reason", ""),
                "ica_mode": args.ica,
                "ica_ran": bool(ica_ran),
                "ica_applied": bool(ica_applied),
                "ica_exclude": " ".join(map(str, ica_exclude)) if ica_exclude else "",
                **{f"ica_fit_{k}": v for k, v in ica_fit_diag.items()},
                **{f"ica_find_{k}": v for k, v in ica_find_diag.items()},
                "review_flag": review_flag,
                "review_reasons": "+".join(review_reasons),
                "status": "OK",
                "error": "",
            }
        )

        print(
            format_alignment_diag(diag, len(markers_aligned))
        )
        print(
            f"Dropped {n_before - n_after}/{n_before} epochs "
            f"(blink={int(blink_bad.sum())}, muscle={int(muscle_bad.sum())})"
        )
        print(
            f"ICA recommended: {ica_recommendation.get('ica_recommended', False)} "
            f"({ica_recommendation.get('ica_recommend_reason', '')})"
        )

    if not any(evokeds_by_cond.values()):
        print("\n[WARN] No successful subjects to grand-average. Writing QC summary only.")
        write_qc_summary(rows, out_dir / "qc_summary.csv")
        parquet_writer.close()

        print(f"Saved QC summary -> {out_dir / 'qc_summary.csv'}")
        return

    ga_by_cond = grand_averages(evokeds_by_cond)
    for cond, ga in ga_by_cond.items():
        ga.save(d_ga / f"grand_average_{cond}-ave.fif", overwrite=True)

    write_qc_summary(rows, out_dir / "qc_summary.csv")
    parquet_writer.close()

    print(f"\nSaved QC summary -> {out_dir / 'qc_summary.csv'}")
    print(f"Saved grand averages -> {d_ga}")


def prepare_output_dirs(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in [
        "01_clean_raw",
        "02_epochs",
        "03_evokeds",
        "04_grand_averages",
        "05_metrics",
        "00_ica",
    ]:
        (out_dir / sub).mkdir(exist_ok=True)

