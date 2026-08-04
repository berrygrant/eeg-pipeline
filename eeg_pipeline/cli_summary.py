from __future__ import annotations

from pathlib import Path

import mne
import pandas as pd

from .cli_common import (
    _behavior_inputs_for_recording,
    _finalize_runtime_paths,
    _recording_from_raw_path,
    align_marker_positions_to_codes,
    compute_ica_diagnostics,
    detect_trigger_bursts,
    events_from_annotations_positions,
    keep_by_gap_heuristic,
    load_behavioral_events,
    marker_gap_stats,
    parse_token_map,
    parse_vmrk_markers,
    read_raw_preprocess,
    recommend_ica,
)


def summarize_one_file(args, raw_path: Path):
    _finalize_runtime_paths(args)
    recording = _recording_from_raw_path(args, raw_path)
    subj = recording.subject_label
    is_bv = raw_path.suffix.lower() == ".vhdr"
    vmrk_path = raw_path.with_suffix(".vmrk") if is_bv else None

    print(f"\n=== SUMMARY: {subj} ===")
    print("Raw file:", raw_path)
    if recording.behavior_kind == "bids_events":
        print("BIDS events:", recording.behavior_path)
    elif recording.behavior_kind == "csv":
        print("Legacy behavior CSV:", recording.behavior_path)
    else:
        print("Behavior source:", recording.behavior_path)
    if is_bv:
        print("VMRK file:", vmrk_path)

    # Show annotation descriptions without any preprocessing (debug)
    if is_bv:
        raw0 = mne.io.read_raw_brainvision(raw_path, preload=True)
    else:
        raw0 = mne.io.read_raw_eeglab(raw_path, preload=True)
    descs = list(dict.fromkeys(raw0.annotations.description))
    print("\nAnnotation descriptions (first 30 unique):")
    print(descs[:30])
    print("Unique annotation count:", len(set(raw0.annotations.description)))

    # Preprocess (montage/reference/filter)
    raw = read_raw_preprocess(
        raw_path=raw_path,
        montage=args.montage,
        eog_chs=args.eog_chs,
        aux_chs=args.aux_chs,
        reref=args.reref,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch=args.notch,
        n_jobs=int(getattr(args, "n_jobs", 1) or 1),
    )

    # ICA diagnostics (non-destructive)
    ica_diag = compute_ica_diagnostics(
        raw,
        blink_proxy_chs=args.blink_proxy_chs,
        blink_threshold_uv=args.blink_threshold_uv,
        blink_win_ms=args.blink_win_ms,
        blink_step_ms=args.blink_step_ms,
    )
    print("\nICA diagnostics:")
    print(pd.Series(ica_diag).to_string())

    # If no true EOG rate, recommend_ica can use proxy
    recommend_ica(
        epoch_reject_rate=0.0,  # unknown yet
        eog_corr_max=ica_diag.get("eog_corr_max", 0.0),
        blink_rate_per_min=ica_diag.get("blink_rate_per_min", 0.0),
        blink_proxy_rate_per_min=ica_diag.get("blink_proxy_rate_per_min", 0.0),
        epoch_loss_thresh=0.20,                 # won’t trigger since 0.0
        eog_corr_thresh=args.ica_corr_thresh,
        blink_rate_thresh=args.ica_auto_blink_rate_per_min,
    )

    events_ann = events_from_annotations_positions(raw)
    markers_pos = events_ann[:, 0].copy()

    # ------------------------------------------------------------
    # StimTrak trigger QC: detect burst-like trigger failures
    # (do NOT modify markers; flag only)
    # ------------------------------------------------------------
    burst_diag = detect_trigger_bursts(
        markers_pos=markers_pos,
        sfreq=float(raw.info["sfreq"]),
        min_iti_s=0.02,      # 20 ms: impossible for real trials
        burst_win_s=0.25,    # 250 ms window
        burst_count=5,       # ≥5 triggers in 250 ms
    )

    if burst_diag["burst_flag"]:
        print(f"[WARN] Trigger burst detected for {subj}: "
              f"short_iti={burst_diag['n_short_iti']}, "
              f"max_in_window={burst_diag['burst_max_in_window']}")
        print("\nTotal events (from annotations):", len(events_ann))
        print("Event ID distribution (from annotations):")
        print(pd.Series(events_ann[:, 2]).value_counts().sort_index().to_string())

    stats = marker_gap_stats(markers_pos, sfreq=float(raw.info["sfreq"]))
    print("\nInter-marker gap stats (seconds):")
    for k in ["dt_min", "dt_p25", "dt_p50", "dt_p75", "dt_p90", "dt_p95", "dt_p99", "dt_max"]:
        if k in stats:
            print(f"  {k}: {stats[k]:.4f}")

    cand_gaps = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    print("\nKeep counts for candidate --drop_eeg_markers_by_gap_s values:")
    for g in cand_gaps:
        keep_idx = keep_by_gap_heuristic(markers_pos, sfreq=float(raw.info["sfreq"]), gap_s=g)
        print(f"  gap_s={g:>4}: keep {len(keep_idx)}/{len(markers_pos)}")

    # Parse .vmrk if present (debug)
    if is_bv and vmrk_path and vmrk_path.exists():
        mk = parse_vmrk_markers(vmrk_path)
        print("\nMarkers from .vmrk:")
        print("  total markers:", len(mk))
        if len(mk):
            print("  marker types:\n", mk["mtype"].value_counts().to_string())
            print("  unique desc count:", mk["desc"].nunique())
            print("  desc distribution (top 10):\n", mk["desc"].value_counts().head(10).to_string())
    elif is_bv:
        print("\n[WARN] .vmrk file not found next to .vhdr; cannot parse markers directly.")

    token_map = parse_token_map(args.token_map)
    csv_fallback_dir = (
        None
        if getattr(args, "behavior_csv_fallback_dir", None) in (None, "")
        else Path(args.behavior_csv_fallback_dir)
    )
    events_tsv, events_json, csv_path, _ = _behavior_inputs_for_recording(recording)
    try:
        behavior = load_behavioral_events(
            events_tsv=events_tsv,
            events_json=events_json,
            subject_id=recording.subject_id,
            keep_codes=args.behavioral_keep_codes,
            token_map=token_map,
            condition_map=getattr(args, "condition_map", None),
            csv_path=csv_path,
            csv_fallback_dir=csv_fallback_dir,
        )
    except FileNotFoundError as exc:
        print("\n[WARN]", str(exc))
        print("Cannot summarize behavioral events without source events or an explicit CSV fallback. Exiting summary.")
        return

    codes_all = behavior.codes_all
    print("\nBehavioral codes (EventCode) count:", len(codes_all))
    print("Behavioral code distribution:")
    print(pd.Series(codes_all).value_counts().sort_index().to_string())

    codes = behavior.codes
    if args.behavioral_keep_codes:
        print("\nBehavioral keep-codes filter applied:")
        print("  keep codes:", list(map(int, args.behavioral_keep_codes)))
        print("  remaining codes:", len(codes))

    print("\nSanity check (Step 4):")
    print("  EEG markers available:", len(markers_pos))
    print("  behavioral codes to assign:", len(codes))

    if behavior.samples is not None:
        aligned = behavior.samples
        diag = {
            "markers_original": int(len(markers_pos)),
            "markers_dropped_by_gap": 0,
            "markers_dropped_by_auto": 0,
        }
        print("  Using BIDS events.tsv sample column directly; EEG marker alignment skipped.")
    else:
        aligned, diag = align_marker_positions_to_codes(
            markers_pos=markers_pos,
            sfreq=float(raw.info["sfreq"]),
            codes=codes,
            gap_s=args.drop_eeg_markers_by_gap_s,
            auto_drop_to_count=bool(args.auto_drop_to_count),
        )
    print("  [OK] alignment achievable.")
    print(
        f"  Alignment: markers {diag['markers_original']} -> {len(aligned)} "
        f"(gap_drop={diag['markers_dropped_by_gap']}, auto_drop={diag['markers_dropped_by_auto']})"
    )

    print("\nToken map:", token_map)
    print("Metadata preview (first 5 rows):")
    print(behavior.metadata.head(5).to_string(index=False))

