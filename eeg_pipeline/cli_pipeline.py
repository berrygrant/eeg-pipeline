# ruff: noqa: F403,F405
from __future__ import annotations

from . import cli_common as _common
from .cli_common import *  # noqa: F403
from .cli_config import apply_config

globals().update({name: value for name, value in vars(_common).items() if not name.startswith("__") or name == "__version__"})


def run_legacy_to_bids_conversion(args, defaults=None, cfg=None) -> list[PipelineRecording]:
    if defaults is None:
        defaults = {}
    if cfg is None:
        cfg = apply_config(args, defaults)

    _finalize_runtime_paths(args, cfg)
    if _input_mode(args, cfg) != "legacy":
        raise ValueError("Legacy-to-BIDS conversion requires --legacy input mode.")

    raw_dir = getattr(args, "raw_dir", None)
    if raw_dir is None:
        raise ValueError("Legacy-to-BIDS conversion requires --raw_dir or paths.raw_dir.")
    bids_root = getattr(args, "conversion_bids_root", None)
    if bids_root is None:
        raise ValueError("Legacy-to-BIDS conversion requires --conversion_bids_root or conversion.bids_output_root.")

    task_label = _legacy_task_label(args, cfg)
    recordings = discover_pipeline_recordings(
        mode="legacy",
        bids_root=None,
        raw_dir=raw_dir,
        subject_csv_dir=getattr(args, "subject_csv_dir", None),
        subjects=getattr(args, "subjects", None),
        sessions=getattr(args, "sessions", None),
        tasks=getattr(args, "tasks", None),
        runs=getattr(args, "runs", None),
        task_label=task_label,
    )
    if not recordings:
        raise RuntimeError(f"No legacy EEG recordings found in {raw_dir}")

    converted = convert_legacy_recordings_to_bids(
        recordings,
        bids_root=bids_root,
        task_label=task_label,
        keep_codes=getattr(args, "behavioral_keep_codes", None),
        standard_codes=getattr(args, "standard_codes", None),
        deviant_codes=getattr(args, "deviant_codes", None),
        drop_eeg_markers_by_gap_s=getattr(args, "drop_eeg_markers_by_gap_s", None),
        auto_drop_to_count=bool(getattr(args, "auto_drop_to_count", 1)),
        overwrite=bool(getattr(args, "conversion_overwrite", 1)),
    )
    args.bids_root = Path(bids_root)
    print(f"Converted legacy dataset -> {args.bids_root}")
    return converted


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
    _finalize_runtime_paths(args, cfg)
    csv_fallback_dir = (
        None
        if getattr(args, "behavior_csv_fallback_dir", None) in (None, "")
        else Path(args.behavior_csv_fallback_dir)
    )

    ep = EpochParams(
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=(float(args.baseline[0]), float(args.baseline[1])),
    )

    token_map = parse_token_map(args.token_map)

    rows: list[dict] = []
    evokeds_by_group: dict[tuple[str | None, str | None], dict[str, list[mne.Evoked]]] = {}

    # Metrics outputs collected across subjects
    erp_metrics_all: list[pd.DataFrame] = []
    tfr_metrics_all: list[pd.DataFrame] = []
    erp_timeseries_all: list[pd.DataFrame] = []

    input_mode = _input_mode(args, cfg)
    task_label = _legacy_task_label(args, cfg)
    source_dataset = getattr(args, "bids_root", None) if input_mode == "bids" else getattr(args, "raw_dir", None)
    if input_mode == "legacy" and bool(getattr(args, "convert_to_bids", False)):
        recordings = run_legacy_to_bids_conversion(args, defaults=defaults, cfg=cfg)
        source_dataset = getattr(args, "bids_root", None)
    else:
        recordings = discover_pipeline_recordings(
            mode=input_mode,
            bids_root=getattr(args, "bids_root", None),
            raw_dir=getattr(args, "raw_dir", None),
            subject_csv_dir=getattr(args, "subject_csv_dir", None),
            subjects=args.subjects,
            sessions=getattr(args, "sessions", None),
            tasks=getattr(args, "tasks", None),
            runs=getattr(args, "runs", None),
            task_label=task_label,
        )
    dataset_root = _prepare_derivatives_root(args, source_dataset=source_dataset)
    _dataset_metrics_dir(dataset_root)
    if not recordings:
        source_root = getattr(args, "bids_root", None) if input_mode == "bids" else getattr(args, "raw_dir", None)
        mode_label = "BIDS" if input_mode == "bids" else "legacy"
        raise RuntimeError(f"No {mode_label} EEG recordings found in {source_root}")

    std_codes = np.asarray(getattr(args, "standard_codes", []) or [], dtype=int)
    dev_codes = np.asarray(getattr(args, "deviant_codes", []) or [], dtype=int)
    stddev_set = np.r_[std_codes, dev_codes]

    condition_map = getattr(args, "condition_map", None)

    metrics_conditions = getattr(args, "metrics_conditions", None)
    if not metrics_conditions:
        if condition_map:
            metrics_conditions = list(condition_map.keys())
        else:
            metrics_conditions = ["Standard", "Deviant"]

    for recording in recordings:
        raw_path = recording.raw_path
        subj = recording.subject_label
        is_bv = raw_path.suffix.lower() == ".vhdr"
        vmrk = raw_path.with_suffix(".vmrk") if is_bv else None
        subject_base = {
            "subject": subj,
            "session": recording.session_label or "",
            "task": recording.task_id or "",
            "run": recording.run_id or "",
            "raw_file": recording.relative_raw_path,
        }
        _, _, _, behavior_hint = _behavior_inputs_for_recording(recording)

        preproc_path = subject_derivative_path(
            dataset_root,
            recording.entities,
            suffix="eeg",
            extension=".fif",
            desc="preproc",
        )
        epochs_path = subject_derivative_path(
            dataset_root,
            recording.entities,
            suffix="epo",
            extension=".fif",
        )
        aligned_events_path = subject_derivative_path(
            dataset_root,
            recording.entities,
            suffix="events",
            extension=".tsv",
            desc="aligned",
        )
        ica_path = subject_derivative_path(
            dataset_root,
            recording.entities,
            suffix="ica",
            extension=".fif",
            desc="components",
        )

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
                            **subject_base,
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
                        **subject_base,
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
        do_ica = False
        if args.ica == "on":
            do_ica = True
        elif args.ica == "auto":
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
                    ica_obj.save(ica_path, overwrite=True)
                    _write_output_sidecar(
                        ica_path,
                        args,
                        recording,
                        behavior_source=behavior_hint,
                        extra={
                            "Description": "ICA solution fit by eeg-pipeline before epoching.",
                            "ICAExclude": list(ica_exclude),
                            "ICAFitDiagnostics": ica_fit_diag,
                            "ICAFindDiagnostics": ica_find_diag,
                        },
                    )

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

        events_tsv, events_json, csv_path, behavior_hint = _behavior_inputs_for_recording(recording)
        try:
            behavior = load_behavioral_events(
                events_tsv=events_tsv,
                events_json=events_json,
                subject_id=recording.subject_id,
                keep_codes=args.behavioral_keep_codes,
                token_map=token_map,
                condition_map=condition_map,
                csv_path=csv_path,
                csv_fallback_dir=csv_fallback_dir,
            )
        except FileNotFoundError as e:
            msg = str(e)
            print("[WARN]", msg, "-> skipping")
            rows.append(
                {
                    **subject_base,
                    **burst_qc,
                    "behavior_source": "missing",
                    "behavior_source_path": str(behavior_hint),
                    "status": "SKIP_MISSING_EVENTS",
                    "error": msg,
                }
            )
            continue

        behavior_source = behavior.source_path
        codes_all = behavior.codes_all
        codes = behavior.codes
        if behavior.samples is not None:
            markers_aligned = np.asarray(behavior.samples, dtype=int)
            diag = {
                "markers_original": int(len(markers_pos)),
                "markers_dropped_by_gap": 0,
                "markers_dropped_by_auto": 0,
            }
        else:
            try:
                markers_aligned, diag = align_marker_positions_to_codes(
                    markers_pos=markers_pos,
                    sfreq=float(raw.info["sfreq"]),
                    codes=codes,
                    gap_s=args.drop_eeg_markers_by_gap_s,
                    auto_drop_to_count=bool(args.auto_drop_to_count),
                )
            except Exception as e:
                msg = f"Alignment failed for {subj}: {e}"
                print("[WARN]", msg, "-> skipping")
                rows.append(
                    {
                        **subject_base,
                        **burst_qc,
                        "behavior_source": behavior.source,
                        "behavior_source_path": str(behavior_source),
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
        if condition_map is None and (len(std_codes) == 0 or len(dev_codes) == 0):
            raise ValueError("Standard and deviant codes are required when no condition_map is provided.")

        if condition_map is None:
            events_stddev, event_id = select_and_recode_stddev(events, args.standard_codes, args.deviant_codes)
        else:
            events_stddev = events
            event_id = {}

        if condition_map is None and len(events_stddev) == 0:
            msg = f"No standard/deviant events after filtering for {subj}"
            print("[WARN]", msg, "-> skipping")
            rows.append(
                {
                    **subject_base,
                    "behavior_source": behavior.source,
                    "behavior_source_path": str(behavior_source),
                    **burst_qc,
                    "status": "SKIP_NO_STDDEV_EVENTS",
                    "error": msg,
                }
            )
            continue

        events_export = _finalized_events_table(
            behavior.metadata,
            sfreq=float(raw.info["sfreq"]),
            samples=markers_aligned,
            codes=codes,
        )
        trial_levels = None
        if "trial_type" in events_export.columns:
            trial_levels = {
                str(value): str(value)
                for value in sorted(events_export["trial_type"].dropna().astype(str).unique())
            }
        _save_dataframe_with_sidecar(
            events_export,
            aligned_events_path,
            args,
            recording,
            behavior_source=behavior_source,
            description="Aligned event table written by eeg-pipeline in BIDS events format.",
            column_descriptions=_events_json_sidecar(events_export, trial_type_levels=trial_levels),
        )

        if condition_map:
            events_epo, event_id, cond_codes = select_and_filter_conditions(events, condition_map)
            keep_mask = np.isin(events[:, 2], np.asarray(cond_codes, dtype=int))
        else:
            events_epo, event_id = select_and_recode_stddev(events, args.standard_codes, args.deviant_codes)
            keep_mask = np.isin(events[:, 2], stddev_set)

        md_full = behavior.metadata.reset_index(drop=True)

        if len(events_epo) == 0:
            reason = "condition_map" if condition_map else "standard/deviant codes"
            msg = f"No matching events after applying {reason}; skipping."
            print("[WARN]", msg)
            rows.append(
                {
                    **subject_base,
                    "behavior_source": behavior.source,
                    "behavior_source_path": str(behavior_source),
                    **diag,
                    **burst_qc,
                    "n_events_used": int(len(events)),
                    "n_events_kept_stddev": 0,
                    "status": "SKIP_NO_CONDITION_EVENTS",
                    "error": msg,
                }
            )
            raw.save(preproc_path, overwrite=True)
            _write_output_sidecar(
                preproc_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={"Description": "Preprocessed continuous EEG after filtering and rereferencing."},
            )
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
        if blink_auto_pct is not None and len(blink_picks) > 0:
            blink_data = epochs_test.get_data(picks=blink_picks)
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
                epochs_test.get_data(picks=blink_picks),
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
        if volt_auto_pct is not None and len(eeg_picks) > 0:
            eeg_data = epochs_test.get_data(picks=eeg_picks)
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
        if volt_method == "window_ptp":
            muscle_bad = moving_window_ptp_mask(
                epochs_test.get_data(picks=eeg_picks),
                sfreq=float(epochs_test.info["sfreq"]),
                win_ms=float(getattr(args, "volt_win_ms", 200.0)),
                step_ms=float(getattr(args, "volt_step_ms", 10.0)),
                threshold_uv=volt_threshold_uv,
            )
        elif volt_method == "combined":
            simple_bad = simple_voltage_threshold_mask(
                epochs_test.get_data(picks=eeg_picks),
                pos_limit_uv=volt_pos_uv,
                neg_limit_uv=volt_neg_uv,
            )
            ptp_bad = moving_window_ptp_mask(
                epochs_test.get_data(picks=eeg_picks),
                sfreq=float(epochs_test.info["sfreq"]),
                win_ms=float(getattr(args, "volt_win_ms", 200.0)),
                step_ms=float(getattr(args, "volt_step_ms", 10.0)),
                threshold_uv=volt_threshold_uv,
            )
            muscle_bad = simple_bad | ptp_bad
        else:
            muscle_bad = simple_voltage_threshold_mask(
                epochs_test.get_data(picks=eeg_picks),
                pos_limit_uv=volt_pos_uv,
                neg_limit_uv=volt_neg_uv,
            )

        step_thresh = getattr(args, "volt_step_uv_per_ms", None)
        if step_thresh not in (None, "None", "null"):
            step_bad = step_threshold_mask(
                epochs_test.get_data(picks=eeg_picks),
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
                    **subject_base,
                    "behavior_source": behavior.source,
                    "behavior_source_path": str(behavior_source),
                    **diag,
                    **burst_qc,
                    **threshold_info,
                    "n_epochs_before_artifact": int(n_before),
                    "n_epochs_final": 0,
                    "status": "SKIP_EMPTY_EPOCHS",
                    "error": msg,
                }
            )
            raw.save(preproc_path, overwrite=True)
            _write_output_sidecar(
                preproc_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={"Description": "Preprocessed continuous EEG after filtering and rereferencing."},
            )
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
                    **subject_base,
                    "behavior_source": behavior.source,
                    "behavior_source_path": str(behavior_source),
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
            raw.save(preproc_path, overwrite=True)
            epochs.save(epochs_path, overwrite=True)
            _write_output_sidecar(
                preproc_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={"Description": "Preprocessed continuous EEG after filtering and rereferencing."},
            )
            _write_output_sidecar(
                epochs_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={
                    "Description": "Subject-level epochs after artifact rejection.",
                    "EventID": event_id,
                },
            )
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
                    **subject_base,
                    "behavior_source": behavior.source,
                    "behavior_source_path": str(behavior_source),
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
            raw.save(preproc_path, overwrite=True)
            epochs.save(epochs_path, overwrite=True)
            _write_output_sidecar(
                preproc_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={"Description": "Preprocessed continuous EEG after filtering and rereferencing."},
            )
            _write_output_sidecar(
                epochs_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={
                    "Description": "Subject-level epochs after artifact rejection.",
                    "EventID": event_id,
                },
            )
            continue

        # ------------------------------------------------------------------
        # Metrics (ERP + TFR)
        # ------------------------------------------------------------------
        if int(getattr(args, "metrics", 0)):
            do_erp = bool(getattr(args, "metrics_erp_enabled", True))
            do_tfr = bool(getattr(args, "metrics_tfr_enabled", True))

            if do_erp or do_tfr:
                subject_metrics_dir = _subject_metrics_dir(dataset_root, recording)
                subject_metrics_dir.mkdir(parents=True, exist_ok=True)

            channels = getattr(args, "metrics_channels", None) or ["Fp1", "Fz", "Cz"]
            conds = metrics_conditions

            if do_erp:
                # ERP windows
                erp_windows = _build_erp_windows(args)

                try:
                    diff_label = getattr(args, "difference_label", None)
                    df_erp = compute_erp_metrics(
                        epochs,
                        subject=subj,
                        channels=channels,
                        conditions=conds,
                        windows=erp_windows,
                        compute_mmn=bool(getattr(args, "compute_mmn", 1)),
                        mmn_name=diff_label if diff_label else "DEV_MINUS_STD",
                    )
                    df_erp["subject"] = subj
                    df_erp["task"] = recording.task_id or ""
                    df_erp["session"] = recording.session_label or ""
                    df_erp["run"] = recording.run_id or ""
                    _save_dataframe_with_sidecar(
                        df_erp,
                        subject_derivative_path(
                            dataset_root,
                            recording.entities,
                            suffix="metrics",
                            extension=".tsv",
                            desc="erp",
                        ),
                        args,
                        recording,
                        behavior_source=behavior_source,
                        description="Subject-level ERP metrics computed from derivative epochs.",
                    )
                    erp_metrics_all.append(df_erp)
                except Exception as e:
                    print(f"[WARN] ERP metrics failed for {subj}: {e}")

                if bool(getattr(args, "metrics_erp_timeseries", False)):
                    try:
                        if args.metrics_channels is None:
                            eeg_picks = mne.pick_types(epochs.info, eeg=True, eog=False)
                            ts_channels = [epochs.ch_names[i] for i in eeg_picks]
                        else:
                            ts_channels = channels

                        ts_params = ERPTimeSeriesParams(
                            tmin=float(args.tmin),
                            tmax=float(args.tmax),
                            baseline=(float(args.baseline[0]), float(args.baseline[1])),
                            decim=1,
                        )
                        df_ts = compute_erp_timeseries(
                            epochs,
                            subject=subj,
                            channels=ts_channels,
                            params=ts_params,
                            conditions=conds,
                            include_difference_wave=False,
                        )
                        df_ts["subject"] = subj
                        df_ts["task"] = recording.task_id or ""
                        df_ts["session"] = recording.session_label or ""
                        df_ts["run"] = recording.run_id or ""
                        _save_dataframe_with_sidecar(
                            df_ts,
                            subject_derivative_path(
                                dataset_root,
                                recording.entities,
                                suffix="timeseries",
                                extension=".parquet",
                                desc="erp",
                            ),
                            args,
                            recording,
                            behavior_source=behavior_source,
                            description="Subject-level ERP time series metrics.",
                        )
                        erp_timeseries_all.append(df_ts)
                    except Exception as e:
                        print(f"[WARN] ERP timeseries failed for {subj}: {e}")

            if do_tfr:
                try:
                    tfr_params = TFRParams(
                        fmin=float(getattr(args, "tfr_fmin", 1.0)),
                        fmax=float(getattr(args, "tfr_fmax", 30.0)),
                        fstep=float(getattr(args, "tfr_fstep", 1.0)),
                        method=str(getattr(args, "tfr_method", "multitaper")),
                        n_cycles_div=float(getattr(args, "tfr_n_cycles_div", 10.0)),
                        decim=int(getattr(args, "tfr_decim", 1)),
                        baseline=(
                            float(getattr(args, "tfr_baseline", [-0.1, 0.0])[0]),
                            float(getattr(args, "tfr_baseline", [-0.1, 0.0])[1]),
                        ),
                        mode=str(getattr(args, "tfr_baseline_mode", "logratio")),
                    )
                    df_tfr = compute_tfr_metrics(
                        epochs,
                        subject=subj,
                        channels=channels,
                        conditions=conds,
                        params=tfr_params,
                        tmin=float(getattr(args, "tfr_tmin", -0.2)),
                        tmax=float(getattr(args, "tfr_tmax", 0.6)),
                        time_decim=int(getattr(args, "tfr_time_decim", 1)),
                    )
                    df_tfr["subject"] = subj
                    df_tfr["task"] = recording.task_id or ""
                    df_tfr["session"] = recording.session_label or ""
                    df_tfr["run"] = recording.run_id or ""
                    _save_dataframe_with_sidecar(
                        df_tfr,
                        subject_derivative_path(
                            dataset_root,
                            recording.entities,
                            suffix="metrics",
                            extension=".tsv",
                            desc="tfr",
                        ),
                        args,
                        recording,
                        behavior_source=behavior_source,
                        description="Subject-level TFR metrics computed from derivative epochs.",
                    )
                    tfr_metrics_all.append(df_tfr)
                except Exception as e:
                    print(f"[WARN] TFR metrics failed for {subj}: {e}")

        ica_recommendation = recommend_ica(
            epoch_reject_rate=epoch_reject_rate,
            eog_corr_max=ica_diag.get("eog_corr_max", 0.0),
            blink_rate_per_min=ica_diag.get("blink_rate_per_min", 0.0),
            blink_proxy_rate_per_min=ica_diag.get("blink_proxy_rate_per_min", 0.0),
            epoch_loss_thresh=0.20,
            eog_corr_thresh=args.ica_corr_thresh,
            blink_rate_thresh=args.ica_auto_blink_rate_per_min,
        )

        raw.save(preproc_path, overwrite=True)
        epochs.save(epochs_path, overwrite=True)
        _write_output_sidecar(
            preproc_path,
            args,
            recording,
            behavior_source=behavior_source,
            extra={
                "Description": "Preprocessed continuous EEG after filtering and rereferencing.",
                "ICAApplied": bool(ica_applied),
            },
        )
        _write_output_sidecar(
            epochs_path,
            args,
            recording,
            behavior_source=behavior_source,
            extra={
                "Description": "Subject-level epochs after artifact rejection.",
                "EventID": event_id,
                "EpochRejectRate": float(epoch_reject_rate),
            },
        )

        evoked_conditions = list(event_id.keys())
        evokeds = compute_evokeds(epochs, evoked_conditions)
        for cond, ev in evokeds.items():
            evoked_path = subject_derivative_path(
                dataset_root,
                recording.entities,
                suffix="ave",
                extension=".fif",
                desc=cond.lower(),
            )
            ev.save(evoked_path, overwrite=True)
            _write_output_sidecar(
                evoked_path,
                args,
                recording,
                behavior_source=behavior_source,
                extra={
                    "Description": f"Subject-level evoked average for {cond}.",
                    "Condition": cond,
                    "Nave": getattr(ev, "nave", None),
                },
            )
            group_key = (recording.session_id, recording.task_id)
            evokeds_by_group.setdefault(group_key, {}).setdefault(cond, []).append(ev)

        rows.append(
            {
                **subject_base,
                "sfreq": float(raw.info["sfreq"]),
                "token1": token_map.get("token1"),
                "token2": token_map.get("token2"),
                "behavior_source": behavior.source,
                "behavior_source_path": str(behavior_source),
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
            f"Alignment: markers {diag['markers_original']} -> {len(markers_aligned)} "
            f"(gap_drop={diag['markers_dropped_by_gap']}, auto_drop={diag['markers_dropped_by_auto']})"
        )
        print(
            f"Dropped {n_before - n_after}/{n_before} epochs "
            f"(blink={int(blink_bad.sum())}, muscle={int(muscle_bad.sum())})"
        )
        print(
            f"ICA recommended: {ica_recommendation.get('ica_recommended', False)} "
            f"({ica_recommendation.get('ica_recommend_reason', '')})"
        )

    qc_path = dataset_derivative_path(
        dataset_root,
        suffix="qc",
        extension=".tsv",
        desc="summary",
    )
    write_qc_summary(rows, qc_path)
    write_json(
        derivative_sidecar_path(qc_path),
        {
            "Description": "Dataset-level QC summary for eeg-pipeline derivatives.",
            "GeneratedBy": [{"Name": PIPELINE_NAME, "Version": __version__}],
        },
    )

    if erp_metrics_all:
        _save_dataframe_with_sidecar(
            pd.concat(erp_metrics_all, ignore_index=True),
            dataset_derivative_path(dataset_root, suffix="metrics", extension=".tsv", desc="erp"),
            args,
            None,
            behavior_source=None,
            description="Dataset-level ERP metrics aggregated across processed subjects.",
        )
    if erp_timeseries_all:
        _save_dataframe_with_sidecar(
            pd.concat(erp_timeseries_all, ignore_index=True),
            dataset_derivative_path(dataset_root, suffix="timeseries", extension=".parquet", desc="erp"),
            args,
            None,
            behavior_source=None,
            description="Dataset-level ERP time series aggregated across processed subjects.",
        )
    if tfr_metrics_all:
        _save_dataframe_with_sidecar(
            pd.concat(tfr_metrics_all, ignore_index=True),
            dataset_derivative_path(dataset_root, suffix="metrics", extension=".tsv", desc="tfr"),
            args,
            None,
            behavior_source=None,
            description="Dataset-level TFR metrics aggregated across processed subjects.",
        )

    if not any(evokeds_by_group.values()):
        print("\n[WARN] No successful subjects to grand-average. Writing QC summary only.")
        print(f"Saved QC summary -> {qc_path}")
        return

    for (ses, task), evoked_map in evokeds_by_group.items():
        ga_by_cond = grand_averages(evoked_map)
        group_entities = {}
        if ses:
            group_entities["ses"] = ses
        if task:
            group_entities["task"] = task
        for cond, ga in ga_by_cond.items():
            ga_path = dataset_derivative_path(
                dataset_root,
                entities=group_entities,
                suffix="ave",
                extension=".fif",
                desc=f"grandaverage-{cond.lower()}",
            )
            ga.save(ga_path, overwrite=True)
            write_json(
                derivative_sidecar_path(ga_path),
                {
                    "Description": f"Grand-average evoked response for {cond}.",
                    "GeneratedBy": [{"Name": PIPELINE_NAME, "Version": __version__}],
                    "Session": ses,
                    "Task": task,
                    "Condition": cond,
                },
            )

    print(f"\nSaved QC summary -> {qc_path}")
    print(f"Saved derivatives -> {dataset_root}")

