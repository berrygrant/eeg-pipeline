# mmn_pipeline/cli.py
from __future__ import annotations

import argparse
import sys
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

# Helper for config integration.  When merging configuration values
# into command‑line arguments we want to honour user‑supplied flags.
# This helper sets ``args.<field>`` only if the attribute still has
# its argparse default value.  The ``defaults`` dict is built once in
# ``main`` and passed through to ``run_full_pipeline``.  See
# ``build_defaults`` below for how defaults are collected.
def set_if_default(args, defaults: dict, field: str, value):
    """Assign a config value to ``args.field`` only if the CLI
    argument was not explicitly provided.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments namespace.  The field is looked up on this
        object and replaced if it still equals the parser default.
    defaults : dict
        Mapping of argument names to their argparse defaults.  This
        comes from ``build_defaults`` which iterates over all parser
        actions.
    field : str
        Name of the attribute on ``args`` to inspect and potentially
        overwrite.
    value : Any
        The value from the configuration that should be applied if
        appropriate.
    """
    # If the field wasn't in defaults we can't know the default so bail
    if field not in defaults:
        return
    # Only set the value if the user didn't override it via CLI
    if getattr(args, field) == defaults[field]:
        setattr(args, field, value)
from eeg_pipeline.config import load_config

# Metrics (ERP + TFR)
from eeg_pipeline.metrics import compute_erp_metrics, compute_tfr_metrics, load_epochs
from eeg_pipeline.metrics.erp import ERPWindow
from eeg_pipeline.metrics.erp_timeseries import ERPTimeSeriesParams, compute_erp_timeseries
from eeg_pipeline.metrics.tfr import TFRParams

import re
import tempfile

_BV_KEY_RE = re.compile(r"^(?P<key>\w+)\s*=\s*(?P<val>.+?)\s*$", re.MULTILINE)

def _bv_get(txt: str, key: str) -> str | None:
    key_l = key.lower()
    for m in _BV_KEY_RE.finditer(txt):
        if m.group("key").strip().lower() == key_l:
            return m.group("val").strip()
    return None

def brainvision_links_ok(vhdr_path: Path) -> tuple[bool, str]:
    """
    Returns (ok, reason). Checks whether .vhdr's MarkerFile/DataFile exist.
    """
    txt = vhdr_path.read_text(encoding="utf-8", errors="replace")
    marker = _bv_get(txt, "MarkerFile")
    data = _bv_get(txt, "DataFile")

    missing = []
    if marker:
        if not (vhdr_path.parent / marker).exists():
            missing.append(f"MarkerFile={marker}")
    if data:
        if not (vhdr_path.parent / data).exists():
            missing.append(f"DataFile={data}")

    if missing:
        return False, "Missing referenced file(s): " + ", ".join(missing)
    return True, ""

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
        reref=args.reref,
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

    # If no true EOG rate, recommend_ica can use proxy
    pre_epoch_reco = recommend_ica(
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

    trigger_diag = {
    "trigger_burst_flag": burst_diag["burst_flag"],
    "trigger_n_short_iti": burst_diag["n_short_iti"],
    "trigger_min_iti_s": burst_diag["min_iti_s"],
    "trigger_burst_max_in_window": burst_diag["burst_max_in_window"],
    "trigger_burst_n_windows_ge_thresh": burst_diag["burst_n_windows_ge_thresh"],
    "trigger_burst_params": burst_diag.get("burst_params", ""),
    }

    burst_qc = {
        "trigger_burst_flag": bool(burst_diag.get("burst_flag", False)),
        "trigger_n_short_iti": int(burst_diag.get("n_short_iti", 0) or 0),
        "trigger_min_iti_s": burst_diag.get("min_iti_s", ""),
        "trigger_burst_max_in_window": int(burst_diag.get("burst_max_in_window", 1) or 1),
        "trigger_burst_n_windows_ge_thresh": int(burst_diag.get("burst_n_windows_ge_thresh", 0) or 0),
        "trigger_burst_params": burst_diag.get("burst_params", ""),
    }

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

def detect_trigger_bursts(markers_pos: np.ndarray, sfreq: float,
                          min_iti_s: float = 0.02,
                          burst_win_s: float = 0.25,
                          burst_count: int = 5) -> dict:
    """
    Detect suspicious StimTrak behavior:
      - very short ITIs (<= min_iti_s)
      - bursts: >= burst_count triggers inside burst_win_s

    Returns summary diagnostics; does NOT modify markers.
    """
    if len(markers_pos) < 2:
        return {
            "burst_flag": False,
            "n_triggers": int(len(markers_pos)),
            "n_short_iti": 0,
            "min_iti_s": None,
            "burst_max_in_window": 1,
            "burst_n_windows_ge_thresh": 0,
        }

    t = markers_pos / float(sfreq)
    dt = np.diff(t)

    n_short = int(np.sum(dt <= min_iti_s))
    min_iti = float(np.min(dt))

    # Sliding window burst count using two pointers
    j = 0
    burst_max = 1
    n_ge = 0
    for i in range(len(t)):
        while t[i] - t[j] > burst_win_s:
            j += 1
        c = i - j + 1
        burst_max = max(burst_max, c)
        if c >= burst_count:
            n_ge += 1

    burst_flag = (n_short > 0) or (burst_max >= burst_count)

    return {
        "burst_flag": bool(burst_flag),
        "n_triggers": int(len(markers_pos)),
        "n_short_iti": int(n_short),
        "min_iti_s": min_iti,
        "burst_max_in_window": int(burst_max),
        "burst_n_windows_ge_thresh": int(n_ge),
        "burst_params": f"min_iti_s={min_iti_s},win_s={burst_win_s},count={burst_count}",
    }


def apply_config(args, defaults=None):
    """Load config and apply values to args (respecting CLI overrides)."""
    if defaults is None:
        defaults = {}
    cfg = load_config(args.config)

    # Paths
    set_if_default(args, defaults, "raw_dir", cfg["paths"]["raw_dir"])
    set_if_default(args, defaults, "subject_csv_dir", cfg["paths"]["subject_csv_dir"])
    set_if_default(args, defaults, "out_dir", cfg["paths"]["out_dir"])

    # Channels and preprocessing
    set_if_default(args, defaults, "montage", cfg["preprocess"].get("montage", args.montage))
    set_if_default(args, defaults, "reref", cfg["preprocess"].get("reref", args.reref))
    set_if_default(args, defaults, "l_freq", cfg["preprocess"].get("l_freq", args.l_freq))
    set_if_default(args, defaults, "h_freq", cfg["preprocess"].get("h_freq", args.h_freq))
    set_if_default(args, defaults, "notch", cfg["preprocess"].get("notch_hz", args.notch))

    # Channel selections
    set_if_default(args, defaults, "eog_chs", cfg["channels"].get("eog_chs", args.eog_chs))
    set_if_default(args, defaults, "blink_proxy_chs", cfg["channels"].get("blink_proxy_chs", args.blink_proxy_chs))
    set_if_default(args, defaults, "aux_chs", cfg["channels"].get("drop_aux_chs", args.aux_chs))

    # Events
    set_if_default(args, defaults, "standard_codes", cfg["events"].get("standard_codes", args.standard_codes))
    set_if_default(args, defaults, "deviant_codes", cfg["events"].get("deviant_codes", args.deviant_codes))
    set_if_default(
        args, defaults, "behavioral_keep_codes",
        cfg["events"].get("behavioral_keep_codes", args.behavioral_keep_codes)
    )
    set_if_default(
        args, defaults, "drop_eeg_markers_by_gap_s",
        cfg["events"].get("drop_eeg_markers_by_gap_s", args.drop_eeg_markers_by_gap_s)
    )
    set_if_default(
        args, defaults, "auto_drop_to_count",
        int(bool(cfg["events"].get("auto_drop_to_count", args.auto_drop_to_count)))
    )

    # Epoching
    set_if_default(args, defaults, "tmin", cfg["epoching"].get("tmin", args.tmin))
    set_if_default(args, defaults, "tmax", cfg["epoching"].get("tmax", args.tmax))
    set_if_default(args, defaults, "baseline", cfg["epoching"].get("baseline", args.baseline))

    # Artifacts
    art = cfg.get("artifacts", {})
    win = art.get("test_window", [args.art_test_tmin, args.art_test_tmax])
    if len(win) >= 2:
        set_if_default(args, defaults, "art_test_tmin", float(win[0]))
        set_if_default(args, defaults, "art_test_tmax", float(win[1]))
    blink_cfg = art.get("blink", {})
    set_if_default(args, defaults, "blink_threshold_uv", blink_cfg.get("threshold_uv", args.blink_threshold_uv))
    set_if_default(args, defaults, "blink_win_ms", blink_cfg.get("win_ms", args.blink_win_ms))
    set_if_default(args, defaults, "blink_step_ms", blink_cfg.get("step_ms", args.blink_step_ms))
    volt_cfg = art.get("voltage", {})
    set_if_default(args, defaults, "volt_pos_uv", volt_cfg.get("pos_uv", args.volt_pos_uv))
    set_if_default(args, defaults, "volt_neg_uv", volt_cfg.get("neg_uv", args.volt_neg_uv))

    # ICA
    ica_cfg = cfg.get("ica", {})
    set_if_default(args, defaults, "ica", ica_cfg.get("mode", args.ica))
    set_if_default(
        args, defaults, "ica_auto_blink_rate_per_min",
        ica_cfg.get("auto_blink_rate_per_min", args.ica_auto_blink_rate_per_min)
    )
    set_if_default(args, defaults, "ica_method", ica_cfg.get("method", args.ica_method))
    set_if_default(args, defaults, "ica_n_components", str(ica_cfg.get("n_components", args.ica_n_components)))
    set_if_default(args, defaults, "ica_random_state", ica_cfg.get("random_state", args.ica_random_state))
    set_if_default(args, defaults, "ica_max_iter", ica_cfg.get("max_iter", args.ica_max_iter))
    set_if_default(args, defaults, "ica_fit_l_freq", ica_cfg.get("fit_l_freq", args.ica_fit_l_freq))
    set_if_default(args, defaults, "ica_fit_h_freq", ica_cfg.get("fit_h_freq", args.ica_fit_h_freq))
    set_if_default(args, defaults, "ica_decim", ica_cfg.get("decim", args.ica_decim))
    set_if_default(args, defaults, "ica_corr_thresh", ica_cfg.get("corr_thresh", args.ica_corr_thresh))
    set_if_default(args, defaults, "ica_max_exclude", ica_cfg.get("max_exclude", args.ica_max_exclude))
    set_if_default(args, defaults, "save_ica", int(bool(ica_cfg.get("save_ica", args.save_ica))))

    # Metrics
    metrics_cfg = cfg.get("metrics", {})
    erp_cfg = metrics_cfg.get("erp", {}) if isinstance(metrics_cfg.get("erp", {}), dict) else {}
    tfr_cfg = metrics_cfg.get("tfr", {}) if isinstance(metrics_cfg.get("tfr", {}), dict) else {}

    erp_enabled = bool(erp_cfg.get("enabled", True))
    tfr_enabled = bool(tfr_cfg.get("enabled", False))

    if "enabled" in metrics_cfg:
        metrics_enabled = bool(metrics_cfg.get("enabled"))
    else:
        metrics_enabled = bool(erp_enabled or tfr_enabled)
    set_if_default(args, defaults, "metrics", int(metrics_enabled))

    # Stash per‑modality enable flags for later gating (no CLI flags)
    args.metrics_erp_enabled = erp_enabled
    args.metrics_tfr_enabled = tfr_enabled
    args.metrics_erp_timeseries = bool(erp_cfg.get("timeseries", False))

    # Only override these from config when the user didn't specify them
    if args.metrics_channels is None:
        chs = erp_cfg.get("channels", None)
        if chs is None:
            chs = metrics_cfg.get("channels", None)
        if isinstance(chs, (list, tuple)) and len(chs):
            args.metrics_channels = list(map(str, chs))

    # ERP windows: list[dict] (preferred) or list[list/tuple]
    if args.erp_window is None:
        wins = erp_cfg.get("windows", None)
        if wins is None:
            wins = metrics_cfg.get("erp_windows", None)
        if isinstance(wins, list) and len(wins):
            parsed = []
            for w in wins:
                if isinstance(w, dict):
                    name = str(w.get("name", "window"))
                    tmin = float(w.get("tmin"))
                    tmax = float(w.get("tmax"))
                    parsed.append([name, tmin, tmax])
                elif isinstance(w, (list, tuple)) and len(w) >= 3:
                    parsed.append([str(w[0]), float(w[1]), float(w[2])])
            if parsed:
                args.erp_window = parsed

    set_if_default(
        args,
        defaults,
        "compute_mmn",
        int(bool(erp_cfg.get("compute_mmn", metrics_cfg.get("compute_mmn", args.compute_mmn)))),
    )

    set_if_default(args, defaults, "tfr_tmin", float(tfr_cfg.get("tmin", args.tfr_tmin)))
    set_if_default(args, defaults, "tfr_tmax", float(tfr_cfg.get("tmax", args.tfr_tmax)))
    set_if_default(args, defaults, "tfr_fmin", float(tfr_cfg.get("fmin", args.tfr_fmin)))
    set_if_default(args, defaults, "tfr_fmax", float(tfr_cfg.get("fmax", args.tfr_fmax)))
    set_if_default(args, defaults, "tfr_fstep", float(tfr_cfg.get("fstep", args.tfr_fstep)))
    set_if_default(args, defaults, "tfr_method", tfr_cfg.get("method", args.tfr_method))
    set_if_default(args, defaults, "tfr_n_cycles_div", float(tfr_cfg.get("n_cycles_div", args.tfr_n_cycles_div)))
    set_if_default(args, defaults, "tfr_decim", int(tfr_cfg.get("decim", args.tfr_decim)))
    set_if_default(args, defaults, "tfr_time_decim", int(tfr_cfg.get("time_decim", args.tfr_time_decim)))
    b = tfr_cfg.get("baseline", [args.tfr_baseline[0], args.tfr_baseline[1]])
    if isinstance(b, (list, tuple)) and len(b) >= 2:
        set_if_default(args, defaults, "tfr_baseline", [float(b[0]), float(b[1])])
    set_if_default(
        args,
        defaults,
        "tfr_baseline_mode",
        tfr_cfg.get("baseline_mode", tfr_cfg.get("mode", args.tfr_baseline_mode)),
    )

    # Token map
    if args.token_map is None:
        tm = cfg.get("labels", {}).get("token_map", None)
        if isinstance(tm, dict):
            args.token_map = [f"{k}={v}" for k, v in tm.items()]
        elif isinstance(tm, list):
            args.token_map = tm
        else:
            args.token_map = None

    return cfg


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

    # Metrics outputs collected across subjects
    erp_metrics_all: list[pd.DataFrame] = []
    tfr_metrics_all: list[pd.DataFrame] = []
    erp_timeseries_all: list[pd.DataFrame] = []

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
        subject_csv_name = subject_csv.name
        subject_csv_path = str(subject_csv)
        subject_csv_exists = bool(subject_csv.exists())
        vmrk = vhdr.with_suffix(".vmrk")

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

        ok, reason = brainvision_links_ok(vhdr)
        if not ok:
            msg = f"BrainVision link mismatch in {vhdr.name}: {reason}"
            if args.on_bv_link_mismatch == "fail":
                raise FileNotFoundError(msg)
            print("[WARN]", msg, "-> skipping")
            rows.append(
                    {
                        "subject": subj,
                        "raw_file": str(vhdr.name),
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
            vhdr_path=vhdr,
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
                        "raw_file": str(vhdr.name),
                        "subject_csv": subject_csv_name,
                        "subject_csv_path": subject_csv_path,
                        "subject_csv_exists": subject_csv_exists,
                        **burst_qc,
                        "status": "SKIP_MISSING_SUBJECT_CSV",
                        "error": msg,
                    }
            )
            continue

        codes_all = read_eventcodes_from_subject_csv(subject_csv)
        codes = filter_codes(codes_all, args.behavioral_keep_codes)
        expected_trials = len(codes) 
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
        epochs = make_epochs(raw, events_stddev, event_id, ep)

        keep_mask = np.isin(events[:, 2], stddev_set)
        md_full = derive_metadata_v1(codes.tolist(), token_map=token_map)
        md = md_full.loc[keep_mask].reset_index(drop=True)
        # Align metadata with epochs that survive MNE's internal dropping
        if len(md) != len(epochs):
            md = md.iloc[epochs.selection].reset_index(drop=True)
        epochs.metadata = md

        epochs_test = epochs.copy().crop(tmin=args.art_test_tmin, tmax=args.art_test_tmax)

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
        n_after = len(epochs)

        if n_after == 0:
            msg = "All epochs dropped after artifact rejection; skipping evoked computation."
            print("[WARN]", msg)
            rows.append(
                    {
                        "subject": subj,
                        "raw_file": str(vhdr.name),
                        "subject_csv": subject_csv_name,
                        "subject_csv_path": subject_csv_path,
                        "subject_csv_exists": subject_csv_exists,
                        **diag,
                        **burst_qc,
                        "n_epochs_before_artifact": int(n_before),
                        "n_epochs_final": 0,
                    "status": "SKIP_EMPTY_EPOCHS",
                    "error": msg,
                }
            )
            raw.save(d_raw / f"{subj}-raw.fif", overwrite=True)
            continue

        n_std = len(epochs["Standard"])
        n_dev = len(epochs["Deviant"])
        if n_std == 0 or n_dev == 0:
            msg = f"Empty condition after rejection (Standard={n_std}, Deviant={n_dev}); skipping evokeds."
            print("[WARN]", msg)
            rows.append(
                    {
                        "subject": subj,
                        "raw_file": str(vhdr.name),
                        "subject_csv": subject_csv_name,
                        "subject_csv_path": subject_csv_path,
                        "subject_csv_exists": subject_csv_exists,
                        **diag,
                        **burst_qc,
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

        # ------------------------------------------------------------------
        # Metrics (ERP + TFR)
        # ------------------------------------------------------------------
        if int(getattr(args, "metrics", 0)):
            do_erp = bool(getattr(args, "metrics_erp_enabled", True))
            do_tfr = bool(getattr(args, "metrics_tfr_enabled", True))

            if do_erp or do_tfr:
                metrics_dir = out_dir / "05_metrics"
                metrics_dir.mkdir(parents=True, exist_ok=True)

            channels = getattr(args, "metrics_channels", None) or ["Fp1", "Fz", "Cz"]

            if do_erp:
                # ERP windows
                if getattr(args, "erp_window", None):
                    erp_windows = [
                        ERPWindow(name=w[0], tmin=float(w[1]), tmax=float(w[2]))
                        for w in args.erp_window
                    ]
                else:
                    erp_windows = [ERPWindow("MMN_150_250", 0.15, 0.25)]

                try:
                    df_erp = compute_erp_metrics(
                        epochs,
                        subject=subj,
                        channels=channels,
                        conditions=["Standard", "Deviant"],
                        windows=erp_windows,
                        compute_mmn=bool(getattr(args, "compute_mmn", 1)),
                    )
                    df_erp.to_csv(metrics_dir / f"{subj}_erp_metrics.csv", index=False)
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
                            conditions=["Standard", "Deviant"],
                            include_difference_wave=False,
                        )
                        ts_dir = metrics_dir / "erp_timeseries"
                        ts_dir.mkdir(parents=True, exist_ok=True)
                        df_ts.to_parquet(ts_dir / f"{subj}_erp_timeseries.parquet", index=False)
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
                        conditions=["Standard", "Deviant"],
                        params=tfr_params,
                        tmin=float(getattr(args, "tfr_tmin", -0.2)),
                        tmax=float(getattr(args, "tfr_tmax", 0.6)),
                        time_decim=int(getattr(args, "tfr_time_decim", 1)),
                    )
                    df_tfr.to_csv(metrics_dir / f"{subj}_tfr_metrics.csv", index=False)
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

    if len(evokeds_std) == 0 or len(evokeds_dev) == 0:
        print("\n[WARN] No successful subjects to grand-average. Writing QC summary only.")
        write_qc_summary(rows, out_dir / "qc_summary.csv")

        # Combined metrics tables (may still exist even if grand averages fail)
        metrics_dir = out_dir / "05_metrics"
        if erp_metrics_all:
            pd.concat(erp_metrics_all, ignore_index=True).to_csv(
                metrics_dir / "erp_metrics_all.csv", index=False
            )
        if erp_timeseries_all:
            pd.concat(erp_timeseries_all, ignore_index=True).to_parquet(
                metrics_dir / "erp_timeseries_all.parquet", index=False
            )
        if tfr_metrics_all:
            pd.concat(tfr_metrics_all, ignore_index=True).to_csv(
                metrics_dir / "tfr_metrics_all.csv", index=False
            )

        print(f"Saved QC summary -> {out_dir / 'qc_summary.csv'}")
        return

    ga_std, ga_dev = grand_averages(evokeds_std, evokeds_dev)
    ga_std.save(d_ga / "grand_average_Standard-ave.fif", overwrite=True)
    ga_dev.save(d_ga / "grand_average_Deviant-ave.fif", overwrite=True)

    write_qc_summary(rows, out_dir / "qc_summary.csv")

    metrics_dir = out_dir / "05_metrics"
    if erp_metrics_all:
        pd.concat(erp_metrics_all, ignore_index=True).to_csv(
            metrics_dir / "erp_metrics_all.csv", index=False
        )
    if erp_timeseries_all:
        pd.concat(erp_timeseries_all, ignore_index=True).to_parquet(
            metrics_dir / "erp_timeseries_all.parquet", index=False
        )
    if tfr_metrics_all:
        pd.concat(tfr_metrics_all, ignore_index=True).to_csv(
            metrics_dir / "tfr_metrics_all.csv", index=False
        )

    print(f"\nSaved QC summary -> {out_dir / 'qc_summary.csv'}")
    print(f"Saved grand averages -> {d_ga}")


def _subject_from_epochs_path(p: Path) -> str:
    stem = p.stem
    if stem.endswith("-epo"):
        stem = stem[:-4]
    return stem


def run_metrics_only(args):
    """Compute ERP/TFR metrics from existing epochs in out_dir/02_epochs."""
    out_dir = Path(args.out_dir)
    epochs_dir = out_dir / "02_epochs"
    metrics_dir = out_dir / "05_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(epochs_dir.glob("*-epo.fif"))
    if not files:
        raise RuntimeError(f"No epochs found in {epochs_dir} (expected *-epo.fif).")

    do_erp = bool(getattr(args, "metrics_erp_enabled", True))
    do_tfr = bool(getattr(args, "metrics_tfr_enabled", True))

    if not (do_erp or do_tfr):
        print("[WARN] Metrics requested but both ERP and TFR are disabled in config.")
        return

    channels = getattr(args, "metrics_channels", None) or ["Fp1", "Fz", "Cz"]

    erp_windows = None
    if do_erp:
        if getattr(args, "erp_window", None):
            erp_windows = [
                ERPWindow(name=w[0], tmin=float(w[1]), tmax=float(w[2]))
                for w in args.erp_window
            ]
        else:
            erp_windows = [ERPWindow("MMN_150_250", 0.15, 0.25)]

    tfr_params = None
    if do_tfr:
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

    erp_metrics_all: list[pd.DataFrame] = []
    tfr_metrics_all: list[pd.DataFrame] = []
    erp_timeseries_all: list[pd.DataFrame] = []

    for p in files:
        subj = _subject_from_epochs_path(p)
        loaded = load_epochs(p)
        epochs = loaded.epochs

        if do_erp and erp_windows is not None:
            try:
                df_erp = compute_erp_metrics(
                    epochs,
                    subject=subj,
                    channels=channels,
                    conditions=["Standard", "Deviant"],
                    windows=erp_windows,
                    compute_mmn=bool(getattr(args, "compute_mmn", 1)),
                )
                df_erp.to_csv(metrics_dir / f"{subj}_erp_metrics.csv", index=False)
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
                    conditions=["Standard", "Deviant"],
                    include_difference_wave=False,
                )
                ts_dir = metrics_dir / "erp_timeseries"
                ts_dir.mkdir(parents=True, exist_ok=True)
                df_ts.to_parquet(ts_dir / f"{subj}_erp_timeseries.parquet", index=False)
                erp_timeseries_all.append(df_ts)
            except Exception as e:
                print(f"[WARN] ERP timeseries failed for {subj}: {e}")

        if do_tfr and tfr_params is not None:
            try:
                df_tfr = compute_tfr_metrics(
                    epochs,
                    subject=subj,
                    channels=channels,
                    conditions=["Standard", "Deviant"],
                    params=tfr_params,
                    tmin=float(getattr(args, "tfr_tmin", -0.2)),
                    tmax=float(getattr(args, "tfr_tmax", 0.6)),
                    time_decim=int(getattr(args, "tfr_time_decim", 1)),
                )
                df_tfr.to_csv(metrics_dir / f"{subj}_tfr_metrics.csv", index=False)
                tfr_metrics_all.append(df_tfr)
            except Exception as e:
                print(f"[WARN] TFR metrics failed for {subj}: {e}")

    if erp_metrics_all:
        pd.concat(erp_metrics_all, ignore_index=True).to_csv(
            metrics_dir / "erp_metrics_all.csv", index=False
        )
    if erp_timeseries_all:
        pd.concat(erp_timeseries_all, ignore_index=True).to_parquet(
            metrics_dir / "erp_timeseries_all.parquet", index=False
        )
    if tfr_metrics_all:
        pd.concat(tfr_metrics_all, ignore_index=True).to_csv(
            metrics_dir / "tfr_metrics_all.csv", index=False
        )


def _resolve_figure_time_window(args) -> tuple[float, float]:
    if args.figure_time_window is not None:
        return float(args.figure_time_window[0]), float(args.figure_time_window[1])
    if getattr(args, "erp_window", None):
        w = args.erp_window[0]
        return float(w[1]), float(w[2])
    return float(args.tmin), float(args.tmax)


def _resolve_figure_freq_band(args) -> tuple[float, float] | None:
    if args.figure_freq_band is not None:
        return float(args.figure_freq_band[0]), float(args.figure_freq_band[1])
    return float(getattr(args, "tfr_fmin", 1.0)), float(getattr(args, "tfr_fmax", 30.0))


def _prompt_yes_no(msg: str) -> bool:
    if not sys.stdin.isatty():
        return False
    resp = input(msg).strip().lower()
    return resp in {"y", "yes"}


def run_plot_figures(args):
    from eeg_pipeline.viz import paper_figures

    out_dir = Path(args.out_dir)
    metrics_dir = out_dir / "05_metrics"
    fig_dir = Path(args.figures_out_dir) if args.figures_out_dir else out_dir / "figures"

    erp_parquet = metrics_dir / "erp_timeseries_all.parquet"
    tfr_file = metrics_dir / "tfr_metrics_all.csv"

    erp_exists = erp_parquet.exists()
    tfr_exists = tfr_file.exists()

    if not erp_exists and not tfr_exists:
        raise FileNotFoundError(
            f"No metrics found for plotting. Expected {erp_parquet} and/or {tfr_file}."
        )

    time_window = _resolve_figure_time_window(args)
    freq_band = _resolve_figure_freq_band(args) if tfr_exists else None

    argv = [
        "--out_dir", str(fig_dir),
        "--time_window", str(time_window[0]), str(time_window[1]),
    ]
    if erp_exists:
        argv += ["--erp_parquet", str(erp_parquet)]
    if tfr_exists and freq_band is not None:
        argv += [
            "--tfr_file", str(tfr_file),
            "--freq_band", str(freq_band[0]), str(freq_band[1]),
        ]
    if args.figure_diff_heatmap:
        argv.append("--diff_heatmap")
    if args.figure_channels:
        argv += ["--channels", *args.figure_channels]

    paper_figures.main(argv)

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
    ap.add_argument("--process_data", action="store_true", help="Process raw data into epochs/evokeds/QC")
    ap.add_argument("--get_metrics", action="store_true", help="Compute ERP/TFR metrics")
    ap.add_argument("--plot_figures", action="store_true", help="Generate paper-ready figures")
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
    ap.add_argument(
        "--reref",
        default="average",
        choices=["average", "none"],
        help="EEG re-reference mode (average or none).",
    )
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

    ap.add_argument(
        "--on_bv_link_mismatch",
        choices=["skip", "fail"],
        default="skip",
        help="What to do if a .vhdr references a missing MarkerFile/DataFile (default: skip).",
    )

    # --- Metrics controls (ERP + TFR) ---
    ap.add_argument(
        "--metrics",
        type=int,
        default=1,
        help="Compute ERP/TFR metrics and write to out_dir/05_metrics (1=yes,0=no).",
    )
    ap.add_argument(
        "--metrics_channels",
        nargs="+",
        default=None,
        help="Channels used for metrics (default uses config or a small fronto-central set).",
    )
    ap.add_argument(
        "--erp_window",
        nargs=3,
        action="append",
        default=None,
        metavar=("NAME", "TMIN", "TMAX"),
        help="Add an ERP window, e.g. --erp_window MMN_150_250 0.15 0.25. Can be repeated.",
    )
    ap.add_argument(
        "--compute_mmn",
        type=int,
        default=1,
        help="If 1, also compute Deviant-Standard for ERP windows (MMN-style).",
    )

    # TFR settings (kept simple; can be overridden in config)
    ap.add_argument("--tfr_tmin", type=float, default=-0.2)
    ap.add_argument("--tfr_tmax", type=float, default=0.6)
    ap.add_argument("--tfr_fmin", type=float, default=1.0)
    ap.add_argument("--tfr_fmax", type=float, default=30.0)
    ap.add_argument("--tfr_fstep", type=float, default=1.0)
    ap.add_argument("--tfr_method", default="multitaper", choices=["multitaper", "morlet"])
    ap.add_argument("--tfr_n_cycles_div", type=float, default=10.0)
    ap.add_argument("--tfr_decim", type=int, default=1)
    ap.add_argument(
        "--tfr_time_decim",
        type=int,
        default=1,
        help="Downsample TFR time points in metrics output (1 = no downsample).",
    )
    ap.add_argument("--tfr_baseline", nargs=2, type=float, default=[-0.1, 0.0])
    ap.add_argument("--tfr_baseline_mode", default="logratio")

    # --- Figure controls ---
    ap.add_argument("--figure_time_window", nargs=2, type=float, default=None, metavar=("TMIN", "TMAX"))
    ap.add_argument("--figure_freq_band", nargs=2, type=float, default=None, metavar=("FMIN", "FMAX"))
    ap.add_argument("--figure_diff_heatmap", action="store_true", help="Add deviant-standard heatmap")
    ap.add_argument("--figure_channels", nargs="+", default=None, help="Optional channel subset for ERP plots")
    ap.add_argument("--figures_out_dir", default=None, help="Output directory for figures (default: out_dir/figures)")
    return ap


# -----------------------------------------------------------------------------
# Default handling helpers
#
# To allow command‑line flags to override YAML/JSON configuration values
# cleanly, we record the argparse defaults once up front.  See
# ``run_full_pipeline`` for how these defaults are used together with the
# ``set_if_default`` helper.
def build_defaults(parser: argparse.ArgumentParser) -> dict:
    """Return a mapping from argument name to its argparse default.

    The ``defaults`` dict allows us to detect whether the user set a flag
    explicitly on the command line (in which case ``args.<field>`` will
    differ from the default) or left it unspecified (in which case we can
    safely override it with the value from the config file).
    """
    defaults: dict = {}
    for action in parser._actions:
        if action.dest != "help":
            defaults[action.dest] = action.default
    return defaults


def main(argv=None):
    ap = build_arg_parser()
    # Collect defaults before parsing arguments.  These defaults let us
    # distinguish CLI‑provided arguments from those left unspecified.
    defaults = build_defaults(ap)
    args = ap.parse_args(argv)

    if args.summarize_one_file:
        summarize_one_file(args, Path(args.summarize_one_file))
        return

    if not (args.process_data or args.get_metrics or args.plot_figures):
        # Default behavior: process data + metrics
        args.process_data = True
        args.get_metrics = True
        args.plot_figures = False

    # Apply config once for all stages
    cfg = apply_config(args, defaults)

    if args.plot_figures:
        # Ensure ERP time-series is available for plotting
        args.metrics_erp_timeseries = True

    if args.process_data:
        if not args.get_metrics:
            args.metrics = 0
        else:
            args.metrics = 1
        run_full_pipeline(args, defaults=defaults, cfg=cfg)
    elif args.get_metrics:
        run_metrics_only(args)

    if args.plot_figures:
        metrics_dir = Path(args.out_dir) / "05_metrics"
        erp_parquet = metrics_dir / "erp_timeseries_all.parquet"
        tfr_file = metrics_dir / "tfr_metrics_all.csv"

        missing = []
        if not erp_parquet.exists():
            missing.append(str(erp_parquet))
        if not tfr_file.exists():
            missing.append(str(tfr_file))

        if missing:
            print(f"[WARN] Missing figure inputs: {', '.join(missing)}")
            if _prompt_yes_no("Run full pipeline now? [y/N] "):
                args.process_data = True
                args.get_metrics = True
                args.metrics = 1
                run_full_pipeline(args, defaults=defaults, cfg=cfg)
            else:
                print("[WARN] Proceeding with available metrics only.")

        run_plot_figures(args)


if __name__ == "__main__":
    main()
