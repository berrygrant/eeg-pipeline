from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import mne
import pandas as pd

from .align import (
    align_marker_positions_to_codes,
    collapse_marker_bursts,
    detect_trigger_bursts,
    format_alignment_diag,
    keep_by_gap_heuristic,
    marker_gap_stats,
)
from .behavior import (
    clean_eventcodes,
    filter_codes,
    read_eventcodes_from_subject_csv,
    resolve_subject_csv_path,
    subject_number_from_stem,
)
from .io_brainvision import (
    events_from_annotations_positions,
    parse_vmrk_markers,
    read_raw_preprocess,
)
from .schema import derive_metadata_v1, parse_token_map


def summarize_one_file(args: Namespace, raw_path: Path) -> None:
    subj = raw_path.stem
    subj_num = subject_number_from_stem(subj)
    subject_csv = resolve_subject_csv_path(Path(args.subject_csv_dir), subj_num, subj)
    is_bv = raw_path.suffix.lower() == ".vhdr"
    vmrk_path = raw_path.with_suffix(".vmrk") if is_bv else None

    print(f"\n=== SUMMARY: {subj} ===")
    print("Raw file:", raw_path)
    print("Subject CSV:", subject_csv)
    if is_bv:
        print("VMRK file:", vmrk_path)

    if is_bv:
        raw0 = mne.io.read_raw_brainvision(raw_path, preload=True)
    else:
        raw0 = mne.io.read_raw_eeglab(raw_path, preload=True)
    descs = list(dict.fromkeys(raw0.annotations.description))
    print("\nAnnotation descriptions (first 30 unique):")
    print(descs[:30])
    print("Unique annotation count:", len(set(raw0.annotations.description)))

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

    from .ica_diagnostics import compute_ica_diagnostics

    ica_diag = compute_ica_diagnostics(
        raw,
        blink_proxy_chs=args.blink_proxy_chs,
        blink_threshold_uv=args.blink_threshold_uv,
        blink_win_ms=args.blink_win_ms,
        blink_step_ms=args.blink_step_ms,
    )
    print("\nICA diagnostics:")
    print(pd.Series(ica_diag).to_string())

    events_ann = events_from_annotations_positions(raw)
    markers_pos = events_ann[:, 0].copy()

    burst_diag = detect_trigger_bursts(
        markers_pos=markers_pos,
        sfreq=float(raw.info["sfreq"]),
        min_iti_s=0.02,
        burst_win_s=0.25,
        burst_count=5,
    )

    if burst_diag["burst_flag"]:
        print(
            f"[WARN] Trigger burst detected for {subj}: "
            f"short_iti={burst_diag['n_short_iti']}, "
            f"max_in_window={burst_diag['burst_max_in_window']}"
        )
        print("\nTotal events (from annotations):", len(events_ann))
        print("Event ID distribution (from annotations):")
        print(pd.Series(events_ann[:, 2]).value_counts().sort_index().to_string())

    stats = marker_gap_stats(markers_pos, sfreq=float(raw.info["sfreq"]))
    print("\nInter-marker gap stats (seconds):")
    for key in ["dt_min", "dt_p25", "dt_p50", "dt_p75", "dt_p90", "dt_p95", "dt_p99", "dt_max"]:
        if key in stats:
            print(f"  {key}: {stats[key]:.4f}")

    print("\nKeep counts for candidate --drop_eeg_markers_by_gap_s values:")
    for gap_s in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        keep_idx = keep_by_gap_heuristic(markers_pos, sfreq=float(raw.info["sfreq"]), gap_s=gap_s)
        print(f"  gap_s={gap_s:>4}: keep {len(keep_idx)}/{len(markers_pos)}")

    print("\nKeep counts for candidate --collapse_eeg_marker_bursts_s values:")
    for burst_s in [0.01, 0.02, 0.03, 0.05]:
        collapsed, _ = collapse_marker_bursts(
            markers_pos,
            sfreq=float(raw.info["sfreq"]),
            min_iti_s=burst_s,
            keep=args.collapse_eeg_marker_bursts_keep,
        )
        print(f"  burst_s={burst_s:>4}: keep {len(collapsed)}/{len(markers_pos)}")

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

    if not subject_csv.exists():
        print("\n[WARN]", f"Missing subject file for {subj}: {subject_csv}")
        print("Cannot summarize behavioral codes without subject CSV. Exiting summary.")
        return

    codes_raw = read_eventcodes_from_subject_csv(subject_csv)
    print("\nBehavioral codes (EventCode) count:", len(codes_raw))
    print("Behavioral code distribution:")
    print(pd.Series(codes_raw).value_counts().sort_index().to_string())

    codes_all, cleanup_diag = clean_eventcodes(codes_raw, args.eventcode_cleanup)
    if cleanup_diag["eventcode_cleanup_removed"] > 0:
        print("\nEventCode cleanup applied:")
        print("  mode:", cleanup_diag["eventcode_cleanup_mode"])
        print("  removed rows:", cleanup_diag["eventcode_cleanup_removed"])
        print("  affected runs:", cleanup_diag["eventcode_cleanup_runs"])
        print("  remaining codes:", len(codes_all))

    codes = filter_codes(codes_all, args.behavioral_keep_codes)
    if args.behavioral_keep_codes:
        print("\nBehavioral keep-codes filter applied:")
        print("  keep codes:", list(map(int, args.behavioral_keep_codes)))
        print("  remaining codes:", len(codes))

    print("\nSanity check (Step 4):")
    print("  EEG markers available:", len(markers_pos))
    print("  behavioral codes to assign:", len(codes))

    aligned, diag = align_marker_positions_to_codes(
        markers_pos=markers_pos,
        sfreq=float(raw.info["sfreq"]),
        codes=codes,
        gap_s=args.drop_eeg_markers_by_gap_s,
        auto_drop_to_count=bool(args.auto_drop_to_count),
        collapse_bursts_s=args.collapse_eeg_marker_bursts_s,
        collapse_keep=args.collapse_eeg_marker_bursts_keep,
    )
    print("  [OK] alignment achievable.")
    print(f"  {format_alignment_diag(diag, len(aligned))}")

    token_map = parse_token_map(args.token_map)
    metadata = derive_metadata_v1(codes.tolist(), token_map=token_map)
    print("\nToken map:", token_map)
    print("Metadata preview (first 5 rows):")
    print(metadata.head(5).to_string(index=False))
