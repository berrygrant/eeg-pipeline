# mmn_pipeline/cli.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import mne

from .schema import parse_token_map, derive_metadata_v1
from .behavior import read_eventcodes_from_subject_csv, filter_codes
from .io_brainvision import read_raw_preprocess, events_from_annotations_positions, parse_vmrk_markers
from .align import marker_gap_stats, keep_by_gap_heuristic, align_marker_positions_to_codes
from .epoching import EpochParams, build_events_from_positions_and_codes, select_and_recode_stddev, make_epochs
from .artifacts import moving_window_ptp_mask, simple_voltage_threshold_mask
from .evoked import compute_evokeds, grand_averages
from .qc import write_qc_summary
from .ica_diagnostics import compute_ica_diagnostics, recommend_ica
from .ica import ICAParams, fit_ica, find_ica_excludes, apply_ica
from eeg_pipeline.config import load_config


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


def subject_number_from_stem(stem: str) -> str:
    s = stem.strip()
    if s.lower().startswith("s") and s[1:].isdigit():
        return s[1:]
    if s.isdigit():
        return s
    digits = "".join([c for c in s if c.isdigit()])
    if not digits:
        raise ValueError(f"Cannot parse subject number from '{stem}'")
    return digits


def summarize_one_file(args, vhdr_path: Path):
    subj = vhdr_path.stem
    subj_num = subject_number_from_stem(subj)
    subject_csv = Path(args.subject_csv_dir) / f"subject-{subj_num}.csv"
    vmrk_path = vhdr_path.with_suffix(".vmrk")

    print(f"\n=== SUMMARY: {subj} ===")
    print("Raw file:", vhdr_path)
    print("Subject CSV:", subject_csv)
    print("VMRK file:", vmrk_path)

    # Show annotation descriptions without any preprocessing (debug)
    raw0 = mne.io.read_raw_brainvision(vhdr_path, preload=True)
    descs = list(dict.fromkeys(raw0.annotations.description))
    print("\nAnnotation descriptions (first 30 unique):")
    print(descs[:30])
    print("Unique annotation count:", len(set(raw0.annotations.description)))

    # Preprocess (montage/reference/filter)
    raw = read_raw_preprocess(
        vhdr_path=vhdr_path,
        montage=args.montage,
        eog_chs=args.eog_chs,
        aux_chs=args.aux_chs,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch=args.notch,
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

    events_ann = events_from_annotations_positions(raw)
    markers_pos = events_ann[:, 0].copy()

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
    if vmrk_path.exists():
        mk = parse_vmrk_markers(vmrk_path)
        print("\nMarkers from .vmrk:")
        print("  total markers:", len(mk))
        if len(mk):
            print("  marker types:\n", mk["mtype"].value_counts().to_string())
            print("  unique desc count:", mk["desc"].nunique())
            print("  desc distribution (top 10):\n", mk["desc"].value_counts().head(10).to_string())
    else:
        print("\n[WARN] .vmrk file not found next to .vhdr; cannot parse markers directly.")

    # Subject CSV required to complete behavioral summary
    if not subject_csv.exists():
        msg = f"Missing subject file for {subj}: {subject_csv}"
        print("\n[WARN]", msg)
        print("Cannot summarize behavioral codes without subject CSV. Exiting summary.")
        return

    codes_all = read_eventcodes_from_subject_csv(subject_csv)
    print("\nBehavioral codes (EventCode) count:", len(codes_all))
    print("Behavioral code distribution:")
    print(pd.Series(codes_all).value_counts().sort_index().to_string())

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
    )
    print("  [OK] alignment achievable.")
    print(
        f"  Alignment: markers {diag['markers_original']} -> {len(aligned)} "
        f"(gap_drop={diag['markers_dropped_by_gap']}, auto_drop={diag['markers_dropped_by_auto']})"
    )

    token_map = parse_token_map(args.token_map)
    md = derive_metadata_v1(codes.tolist(), token_map=token_map)
    print("\nToken map:", token_map)
    print("Metadata preview (first 5 rows):")
    print(md.head(5).to_string(index=False))


def run_full_pipeline(args):
    cfg = load_config(args.config)

    raw_dir = cfg["paths"]["raw_dir"]
    subject_csv_dir = cfg["paths"]["subject_csv_dir"]
    out_dir = cfg["paths"]["out_dir"]
    prepare_output_dirs(out_dir)


    standard_codes = cfg["events"]["standard_codes"]
    deviant_codes = cfg["events"]["deviant_codes"]
    behavioral_keep_codes = cfg["events"]["behavioral_keep_codes"]

    token_map = cfg["labels"]["token_map"]  # either None or {"token1": "...", "token2": "..."}

    d_raw = out_dir / "01_clean_raw"
    d_epo = out_dir / "02_epochs"
    d_evk = out_dir / "03_evokeds"
    d_ga = out_dir / "04_grand_averages"
    for d in (d_raw, d_epo, d_evk, d_ga):
        d.mkdir(parents=True, exist_ok=True)

    ep = EpochParams(
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=(float(args.baseline[0]), float(args.baseline[1])),
    )

    token_map = parse_token_map(args.token_map)

    rows: list[dict] = []
    evokeds_std = []
    evokeds_dev = []

    vhdr_files = sorted(raw_dir.glob("*.vhdr"))
    if not vhdr_files:
        raise RuntimeError(f"No .vhdr files found in {raw_dir}")

    if args.subjects:
        wanted = {s.lower() for s in args.subjects}
        vhdr_files = [p for p in vhdr_files if p.stem.lower() in wanted]
        if not vhdr_files:
            raise RuntimeError(f"No matching .vhdr files found for --subjects={args.subjects}")

    std_codes = np.asarray(args.standard_codes, dtype=int)
    dev_codes = np.asarray(args.deviant_codes, dtype=int)
    stddev_set = np.r_[std_codes, dev_codes]

    for vhdr in vhdr_files:
        subj = vhdr.stem
        subj_num = subject_number_from_stem(subj)
        subject_csv = subject_csv_dir / f"subject-{subj_num}.csv"
        vmrk = vhdr.with_suffix(".vmrk")

        print(f"\n=== {subj} ===")

        # Optional vmrk policy (debugging / audit)
        if not vmrk.exists():
            msg = f"Missing .vmrk for {subj}: {vmrk}"
            if args.on_missing_vmrk == "fail":
                raise FileNotFoundError(msg)
            if args.on_missing_vmrk == "skip":
                print("[WARN]", msg, "-> skipping")
                rows.append(
                    {
                        "subject": subj,
                        "raw_file": str(vhdr.name),
                        "subject_csv": str(subject_csv),
                        "status": "SKIP_MISSING_VMRK",
                        "error": msg,
                    }
                )
                continue
            print("[WARN]", msg)

        # Preprocess
        raw = read_raw_preprocess(
            vhdr_path=vhdr,
            montage=args.montage,
            eog_chs=args.eog_chs,
            aux_chs=args.aux_chs,
            l_freq=args.l_freq,
            h_freq=args.h_freq,
            notch=args.notch,
        )

        # ICA diagnostics (non-destructive)
        ica_diag = compute_ica_diagnostics(
            raw,
            blink_proxy_chs=args.blink_proxy_chs,
            blink_threshold_uv=args.blink_threshold_uv,
            blink_win_ms=args.blink_win_ms,
            blink_step_ms=args.blink_step_ms,
        )

        # ---- ICA: optional fit + apply (before event extraction / epoching) ----
        ica_applied = False
        ica_exclude: list[int] = []
        ica_fit_diag: dict = {}
        ica_find_diag: dict = {}

        if args.ica in ("on", "auto"):
            do_ica = True
            if args.ica == "auto":
                # Use EOG-derived blink rate if available, else fall back to proxy-derived rate
                rate = float(ica_diag.get("blink_rate_per_min", 0.0) or 0.0)
                if rate == 0.0:
                    rate = float(ica_diag.get("blink_proxy_rate_per_min", 0.0) or 0.0)
                do_ica = rate >= args.ica_auto_blink_rate_per_min

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

                # Save ICA object for audit/reuse
                if bool(args.save_ica):
                    ica_path = out_dir / "00_ica" / f"{subj}-ica.fif"
                    ica_path.parent.mkdir(parents=True, exist_ok=True)
                    ica_obj.save(ica_path, overwrite=True)

        # Events from annotations
        events_ann = events_from_annotations_positions(raw)
        markers_pos = events_ann[:, 0].copy()

        # Behavioral file policy
        if not subject_csv.exists():
            msg = f"Missing subject file for {subj}: {subject_csv}"
            if args.on_missing_subject_csv == "fail":
                raise FileNotFoundError(msg)
            print("[WARN]", msg, "-> skipping")
            rows.append(
                {
                    "subject": subj,
                    "raw_file": str(vhdr.name),
                    "subject_csv": str(subject_csv),
                    "status": "SKIP_MISSING_SUBJECT_CSV",
                    "error": msg,
                }
            )
            continue

        # Load + filter behavioral codes
        codes_all = read_eventcodes_from_subject_csv(subject_csv)
        codes = filter_codes(codes_all, args.behavioral_keep_codes)

        # Align markers to behavioral codes (sanity check step)
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
                    "subject": subj,
                    "raw_file": str(vhdr.name),
                    "subject_csv": str(subject_csv),
                    "status": "SKIP_ALIGNMENT_FAILED",
                    "error": msg,
                }
            )
            continue

        events = build_events_from_positions_and_codes(markers_aligned, codes)

        # Epoch only standard/deviant codes (recode to 1/2)
        events_stddev, event_id = select_and_recode_stddev(events, args.standard_codes, args.deviant_codes)
        epochs = make_epochs(raw, events_stddev, event_id, ep)

        # Attach metadata (derived from aligned codes; slice to std/dev kept)
        keep_mask = np.isin(events[:, 2], stddev_set)
        md_full = derive_metadata_v1(codes.tolist(), token_map=token_map)
        epochs.metadata = md_full.loc[keep_mask].reset_index(drop=True)

        # Artifact rejection within test window
        epochs_test = epochs.copy().crop(tmin=args.art_test_tmin, tmax=args.art_test_tmax)

        # Blink detection: prefer true EOG picks; else fall back to blink_proxy_chs (e.g., Fp1)
        eog_picks = mne.pick_types(epochs_test.info, eog=True, eeg=False)
        blink_bad = np.zeros(len(epochs_test), dtype=bool)

        if len(eog_picks) > 0:
            blink_bad = moving_window_ptp_mask(
                epochs_test.get_data(picks=eog_picks),
                sfreq=float(epochs_test.info["sfreq"]),
                win_ms=args.blink_win_ms,
                step_ms=args.blink_step_ms,
                threshold_uv=args.blink_threshold_uv,
            )
        else:
            proxy = [ch for ch in args.blink_proxy_chs if ch in epochs_test.ch_names]
            if proxy:
                proxy_picks = mne.pick_channels(epochs_test.ch_names, include=proxy)
                blink_bad = moving_window_ptp_mask(
                    epochs_test.get_data(picks=proxy_picks),
                    sfreq=float(epochs_test.info["sfreq"]),
                    win_ms=args.blink_win_ms,
                    step_ms=args.blink_step_ms,
                    threshold_uv=args.blink_threshold_uv,
                )

        eeg_picks = mne.pick_types(epochs_test.info, eeg=True, eog=False)
        muscle_bad = simple_voltage_threshold_mask(
            epochs_test.get_data(picks=eeg_picks),
            pos_limit_uv=args.volt_pos_uv,
            neg_limit_uv=args.volt_neg_uv,
        )

        bad = blink_bad | muscle_bad
        bad_idx = np.where(bad)[0].tolist()

        n_before = len(epochs)
        if bad_idx:
            epochs.drop(bad_idx, reason="ARTIFACT_REJECT_MNE")
        # After dropping bad epochs:
        n_after = len(epochs)

        if n_after == 0:
            msg = "All epochs dropped after artifact rejection; skipping evoked computation."
            print("[WARN]", msg)
            rows.append(
                {
                    "subject": subj,
                    "raw_file": str(vhdr.name),
                    "subject_csv": str(subject_csv.name),
                    "status": "SKIP_EMPTY_EPOCHS",
                    "error": msg,
                    **diag,
                    "n_epochs_before_artifact": int(n_before),
                    "n_epochs_final": 0,
                }
            )
            # still save raw; optionally save empty epochs if you want, but don't compute evokeds
            raw.save(d_raw / f"{subj}-raw.fif", overwrite=True)
            # epochs.save(...) optional; but if you do, it will warn "no data"
            continue

        # Also guard per-condition:
        n_std = len(epochs["Standard"])
        n_dev = len(epochs["Deviant"])
        if n_std == 0 or n_dev == 0:
            msg = f"Empty condition after rejection (Standard={n_std}, Deviant={n_dev}); skipping evokeds."
            print("[WARN]", msg)
            rows.append(
                {
                    "subject": subj,
                    "raw_file": str(vhdr.name),
                    "subject_csv": str(subject_csv.name),
                    "status": "SKIP_EMPTY_CONDITION",
                    "error": msg,
                    **diag,
                    "n_epochs_before_artifact": int(n_before),
                    "n_epochs_final": int(n_after),
                    "n_standard_final": int(n_std),
                    "n_deviant_final": int(n_dev),
                }
            )
            raw.save(d_raw / f"{subj}-raw.fif", overwrite=True)
            epochs.save(d_epo / f"{subj}-epo.fif", overwrite=True)
            continue
        epoch_reject_rate = (n_before - n_after) / n_before if n_before > 0 else 0.0

        ica_recommendation = recommend_ica(
            epoch_reject_rate=epoch_reject_rate,
            eog_corr_max=ica_diag.get("eog_corr_max", 0.0),
            blink_rate_per_min=ica_diag.get("blink_rate_per_min", 0.0),
            blink_proxy_rate_per_min=ica_diag.get("blink_proxy_rate_per_min", 0.0),
        )

        # Save outputs for this subject
        raw.save(d_raw / f"{subj}-raw.fif", overwrite=True)
        epochs.save(d_epo / f"{subj}-epo.fif", overwrite=True)

        evo_std, evo_dev = compute_evokeds(epochs)
        evo_std.save(d_evk / f"{subj}_Standard-ave.fif", overwrite=True)
        evo_dev.save(d_evk / f"{subj}_Deviant-ave.fif", overwrite=True)

        evokeds_std.append(evo_std)
        evokeds_dev.append(evo_dev)

        rows.append(
            {
                "subject": subj,
                "raw_file": str(vhdr.name),
                "subject_csv": str(subject_csv.name),
                "sfreq": float(raw.info["sfreq"]),
                "token1": token_map.get("token1"),
                "token2": token_map.get("token2"),
                "behavioral_codes_total": int(len(codes_all)),
                "behavioral_codes_used": int(len(codes)),
                "behavioral_keep_codes": " ".join(map(str, args.behavioral_keep_codes)) if args.behavioral_keep_codes else "",
                **diag,
                "n_events_used": int(len(events)),
                "n_events_kept_stddev": int(len(events_stddev)),
                "n_epochs_before_artifact": int(n_before),
                "n_blink_bad": int(blink_bad.sum()),
                "n_muscle_bad": int(muscle_bad.sum()),
                "n_epochs_dropped": int(n_before - n_after),
                "n_epochs_final": int(n_after),
                "n_standard_final": int(len(epochs["Standard"])),
                "n_deviant_final": int(len(epochs["Deviant"])),
                "epoch_reject_rate": float(epoch_reject_rate),
                "eog_corr_max": float(ica_diag.get("eog_corr_max", 0.0) or 0.0),
                "eog_corr_mean": float(ica_diag.get("eog_corr_mean", 0.0) or 0.0),
                "blink_rate_per_min": float(ica_diag.get("blink_rate_per_min", 0.0) or 0.0),
                "blink_proxy_rate_per_min": float(ica_diag.get("blink_proxy_rate_per_min", 0.0) or 0.0),
                "blink_source": ica_diag.get("blink_source", ""),
                "ica_recommended": bool(ica_recommendation.get("ica_recommended", False)),
                "ica_recommend_reason": ica_recommendation.get("ica_recommend_reason", ""),
                "ica_mode": args.ica,
                "ica_applied": bool(ica_applied),
                "ica_exclude": " ".join(map(str, ica_exclude)) if ica_exclude else "",
                **{f"ica_fit_{k}": v for k, v in ica_fit_diag.items()},
                **{f"ica_find_{k}": v for k, v in ica_find_diag.items()},
                "status": "OK",
                "error": "",
            }
        )

        print(
            f"Alignment: markers {diag['markers_original']} -> {len(markers_aligned)} "
            f"(gap_drop={diag['markers_dropped_by_gap']}, auto_drop={diag['markers_dropped_by_auto']})"
        )
        print(f"Dropped {n_before - n_after}/{n_before} epochs (blink={int(blink_bad.sum())}, muscle={int(muscle_bad.sum())})")
        print(f"ICA recommended: {ica_recommendation.get('ica_recommended', False)} ({ica_recommendation.get('ica_recommend_reason', '')})")

    # Grand averages (only if we have any successful subjects)
    if len(evokeds_std) == 0 or len(evokeds_dev) == 0:
        print("\n[WARN] No successful subjects to grand-average. Writing QC summary only.")
        write_qc_summary(rows, out_dir / "qc_summary.csv")
        print(f"Saved QC summary -> {out_dir / 'qc_summary.csv'}")
        return

    ga_std, ga_dev = grand_averages(evokeds_std, evokeds_dev)
    ga_std.save(d_ga / "grand_average_Standard-ave.fif", overwrite=True)
    ga_dev.save(d_ga / "grand_average_Deviant-ave.fif", overwrite=True)

    write_qc_summary(rows, out_dir / "qc_summary.csv")
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

def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    ap.add_argument("--raw_dir",  help="Folder containing BrainVision .vhdr files")
    ap.add_argument("--subject_csv_dir",  help="Folder containing subject-###.csv files")
    ap.add_argument("--out_dir", help="Output root folder")
    ap.add_argument("--summarize_one_file", default=None, help="If provided, summarize this .vhdr and exit.")

    ap.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional list of subject stems to run (e.g., S203 s204). If omitted, runs all .vhdr files in raw_dir.",
    )

    ap.add_argument(
        "--on_missing_subject_csv",
        choices=["skip", "fail"],
        default="skip",
        help="What to do if subject-###.csv is missing (default: skip).",
    )

    ap.add_argument(
        "--on_missing_vmrk",
        choices=["warn", "skip", "fail"],
        default="warn",
        help="What to do if .vmrk is missing next to .vhdr (default: warn).",
    )

    ap.add_argument("--montage", default="standard_1020", help="Montage name")
    ap.add_argument("--l_freq", type=float, default=0.1, help="High-pass Hz")
    ap.add_argument("--h_freq", type=float, default=30.0, help="Low-pass Hz")
    ap.add_argument("--notch", type=float, nargs="*", default=[60.0], help="Notch freqs Hz")

    ap.add_argument("--tmin", type=float, default=-0.2, help="Epoch start (s)")
    ap.add_argument("--tmax", type=float, default=0.6, help="Epoch end (s)")
    ap.add_argument("--baseline", type=float, nargs=2, default=(-0.2, 0.0), help="Baseline (s s)")

    ap.add_argument("--eog_chs", nargs="*", default=[], help="EOG channel names (if present)")
    ap.add_argument("--aux_chs", nargs="*", default=["AUX"], help="Aux channels to drop")

    ap.add_argument(
        "--blink_proxy_chs",
        nargs="*",
        default=["Fp1"],
        help="Frontal EEG channels to use as blink proxy if no EOG channels exist (default: Fp1).",
    )

    ap.add_argument(
        "--behavioral_keep_codes",
        nargs="*",
        type=int,
        default=[110, 111, 210, 211],
        help="Keep only these EventCode values from subject-###.csv when aligning to EEG markers.",
    )
    ap.add_argument(
        "--drop_eeg_markers_by_gap_s",
        type=float,
        default=None,
        help="Optional gap threshold heuristic (seconds) to drop likely boundary markers before auto-drop-to-count.",
    )
    ap.add_argument(
        "--auto_drop_to_count",
        type=int,
        default=1,
        help="If EEG markers > behavioral codes used, auto-drop extra markers to match count (1=yes,0=no).",
    )

    ap.add_argument("--standard_codes", nargs="*", type=int, default=[110, 210], help="Codes considered Standard")
    ap.add_argument("--deviant_codes", nargs="*", type=int, default=[111, 211], help="Codes considered Deviant")

    ap.add_argument(
        "--token_map",
        nargs="*",
        default=None,
        help="Optional token labeling. Either: '--token_map EH IH' or '--token_map Token1=EH Token2=IH' (or mix).",
    )

    # Artifact settings
    ap.add_argument("--art_test_tmin", type=float, default=-0.2)
    ap.add_argument("--art_test_tmax", type=float, default=0.3)
    ap.add_argument("--blink_threshold_uv", type=float, default=75.0)
    ap.add_argument("--blink_win_ms", type=float, default=200.0)
    ap.add_argument("--blink_step_ms", type=float, default=10.0)
    ap.add_argument("--volt_pos_uv", type=float, default=150.0)
    ap.add_argument("--volt_neg_uv", type=float, default=-150.0)

    # --- ICA controls ---
    ap.add_argument(
        "--ica",
        choices=["off", "auto", "on"],
        default="off",
        help="ICA mode: off (default), auto (gate by blink rate), or on (always run ICA).",
    )
    ap.add_argument("--ica_method", default="fastica", choices=["fastica", "picard", "infomax"])
    ap.add_argument(
        "--ica_n_components",
        default="0.99",
        type=str,
        help="ICA n_components: float variance fraction (e.g., 0.99) or int (e.g., 20).",
    )
    ap.add_argument("--ica_random_state", default=97, type=int)
    ap.add_argument("--ica_max_iter", default=512, type=int)
    ap.add_argument(
        "--ica_fit_l_freq",
        default=1.0,
        type=float,
        help="High-pass used only for ICA fitting (recommended 1.0).",
    )
    ap.add_argument("--ica_fit_h_freq", default=None, type=float, help="Optional low-pass used only for ICA fitting.")
    ap.add_argument("--ica_decim", default=3, type=int, help="Decimation for ICA fit speed (3 is a good default).")
    ap.add_argument("--ica_corr_thresh", default=0.30, type=float, help="Proxy correlation threshold for excluding components.")
    ap.add_argument("--ica_max_exclude", default=3, type=int, help="Max # components to exclude.")
    ap.add_argument(
        "--ica_auto_blink_rate_per_min",
        default=15.0,
        type=float,
        help="If --ica auto, run ICA when blink rate >= this threshold (per minute).",
    )
    ap.add_argument("--save_ica", default=1, type=int, help="Save ICA object to out_dir/00_ica (1=yes,0=no).")

    return ap


def main(argv=None):
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    if args.summarize_one_file:
        summarize_one_file(args, Path(args.summarize_one_file))
        return

    run_full_pipeline(args)