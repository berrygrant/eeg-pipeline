from __future__ import annotations

from argparse import Namespace
from typing import Any

from .config import load_config


def set_if_default(args: Namespace, defaults: dict, field: str, value: Any) -> None:
    """Assign a config value only when the CLI argument still has its default."""
    if field not in defaults:
        return
    if getattr(args, field) == defaults[field]:
        setattr(args, field, value)


def apply_config(args: Namespace, defaults: dict | None = None) -> dict[str, Any]:
    """Load config and apply values to args while respecting CLI overrides."""
    if defaults is None:
        defaults = {}
    cfg = load_config(args.config)

    set_if_default(args, defaults, "raw_dir", cfg["paths"]["raw_dir"])
    set_if_default(args, defaults, "subject_csv_dir", cfg["paths"]["subject_csv_dir"])
    set_if_default(args, defaults, "out_dir", cfg["paths"]["out_dir"])

    set_if_default(args, defaults, "montage", cfg["preprocess"].get("montage", args.montage))
    set_if_default(args, defaults, "reref", cfg["preprocess"].get("reref", args.reref))
    set_if_default(args, defaults, "l_freq", cfg["preprocess"].get("l_freq", args.l_freq))
    set_if_default(args, defaults, "h_freq", cfg["preprocess"].get("h_freq", args.h_freq))
    set_if_default(args, defaults, "notch", cfg["preprocess"].get("notch_hz", args.notch))

    set_if_default(args, defaults, "eog_chs", cfg["channels"].get("eog_chs", args.eog_chs))
    set_if_default(
        args,
        defaults,
        "blink_proxy_chs",
        cfg["channels"].get("blink_proxy_chs", args.blink_proxy_chs),
    )
    set_if_default(args, defaults, "aux_chs", cfg["channels"].get("drop_aux_chs", args.aux_chs))

    set_if_default(args, defaults, "standard_codes", cfg["events"].get("standard_codes", args.standard_codes))
    set_if_default(args, defaults, "deviant_codes", cfg["events"].get("deviant_codes", args.deviant_codes))
    set_if_default(
        args,
        defaults,
        "behavioral_keep_codes",
        cfg["events"].get("behavioral_keep_codes", args.behavioral_keep_codes),
    )
    set_if_default(
        args,
        defaults,
        "eventcode_cleanup",
        cfg["events"].get("eventcode_cleanup", args.eventcode_cleanup),
    )
    set_if_default(
        args,
        defaults,
        "drop_eeg_markers_by_gap_s",
        cfg["events"].get("drop_eeg_markers_by_gap_s", args.drop_eeg_markers_by_gap_s),
    )
    set_if_default(
        args,
        defaults,
        "collapse_eeg_marker_bursts_s",
        cfg["events"].get("collapse_eeg_marker_bursts_s", args.collapse_eeg_marker_bursts_s),
    )
    set_if_default(
        args,
        defaults,
        "collapse_eeg_marker_bursts_keep",
        cfg["events"].get("collapse_eeg_marker_bursts_keep", args.collapse_eeg_marker_bursts_keep),
    )
    set_if_default(
        args,
        defaults,
        "auto_drop_to_count",
        int(bool(cfg["events"].get("auto_drop_to_count", args.auto_drop_to_count))),
    )
    cond_map = cfg["events"].get("condition_map", None)
    if cond_map is not None:
        args.condition_map = cond_map

    set_if_default(args, defaults, "tmin", cfg["epoching"].get("tmin", args.tmin))
    set_if_default(args, defaults, "tmax", cfg["epoching"].get("tmax", args.tmax))
    set_if_default(args, defaults, "baseline", cfg["epoching"].get("baseline", args.baseline))

    art = cfg.get("artifacts", {})
    win = art.get("test_window", [args.art_test_tmin, args.art_test_tmax])
    if len(win) >= 2:
        set_if_default(args, defaults, "art_test_tmin", float(win[0]))
        set_if_default(args, defaults, "art_test_tmax", float(win[1]))
    blink_cfg = art.get("blink", {})
    set_if_default(args, defaults, "blink_threshold_uv", blink_cfg.get("threshold_uv", args.blink_threshold_uv))
    set_if_default(args, defaults, "blink_win_ms", blink_cfg.get("win_ms", args.blink_win_ms))
    set_if_default(args, defaults, "blink_step_ms", blink_cfg.get("step_ms", args.blink_step_ms))
    set_if_default(args, defaults, "blink_auto_percentile", blink_cfg.get("auto_percentile", args.blink_auto_percentile))
    volt_cfg = art.get("voltage", {})
    set_if_default(args, defaults, "volt_pos_uv", volt_cfg.get("pos_uv", args.volt_pos_uv))
    set_if_default(args, defaults, "volt_neg_uv", volt_cfg.get("neg_uv", args.volt_neg_uv))
    if "volt_method" not in defaults:
        args.volt_method = volt_cfg.get("method", "simple")
    else:
        set_if_default(args, defaults, "volt_method", volt_cfg.get("method", args.volt_method))
    if "volt_threshold_uv" not in defaults:
        args.volt_threshold_uv = volt_cfg.get("threshold_uv", 150.0)
    else:
        set_if_default(args, defaults, "volt_threshold_uv", volt_cfg.get("threshold_uv", args.volt_threshold_uv))
    if "volt_win_ms" not in defaults:
        args.volt_win_ms = volt_cfg.get("win_ms", 200.0)
    else:
        set_if_default(args, defaults, "volt_win_ms", volt_cfg.get("win_ms", args.volt_win_ms))
    if "volt_step_ms" not in defaults:
        args.volt_step_ms = volt_cfg.get("step_ms", 10.0)
    else:
        set_if_default(args, defaults, "volt_step_ms", volt_cfg.get("step_ms", args.volt_step_ms))
    if "volt_step_uv_per_ms" not in defaults:
        args.volt_step_uv_per_ms = volt_cfg.get("step_uv_per_ms", None)
    else:
        set_if_default(args, defaults, "volt_step_uv_per_ms", volt_cfg.get("step_uv_per_ms", args.volt_step_uv_per_ms))
    if "volt_auto_percentile" not in defaults:
        args.volt_auto_percentile = volt_cfg.get("auto_percentile", None)
    else:
        set_if_default(args, defaults, "volt_auto_percentile", volt_cfg.get("auto_percentile", args.volt_auto_percentile))

    if "max_reject_rate" not in defaults:
        args.max_reject_rate = art.get("max_reject_rate", None)
    else:
        set_if_default(args, defaults, "max_reject_rate", art.get("max_reject_rate", args.max_reject_rate))

    ica_cfg = cfg.get("ica", {})
    set_if_default(args, defaults, "ica", ica_cfg.get("mode", args.ica))
    set_if_default(
        args,
        defaults,
        "ica_auto_blink_rate_per_min",
        ica_cfg.get("auto_blink_rate_per_min", args.ica_auto_blink_rate_per_min),
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

    metrics_cfg = cfg.get("metrics", {})
    erp_cfg = metrics_cfg.get("erp", {}) if isinstance(metrics_cfg.get("erp", {}), dict) else {}
    tfr_cfg = metrics_cfg.get("tfr", {}) if isinstance(metrics_cfg.get("tfr", {}), dict) else {}

    erp_enabled = bool(erp_cfg.get("enabled", True))
    tfr_enabled = bool(tfr_cfg.get("enabled", False))
    metrics_enabled = bool(metrics_cfg.get("enabled")) if "enabled" in metrics_cfg else bool(erp_enabled or tfr_enabled)
    set_if_default(args, defaults, "metrics", int(metrics_enabled))

    args.metrics_erp_enabled = erp_enabled
    args.metrics_tfr_enabled = tfr_enabled
    args.metrics_erp_timeseries = bool(erp_cfg.get("timeseries", False))

    if args.metrics_channels is None:
        chs = erp_cfg.get("channels", None)
        if chs is None:
            chs = metrics_cfg.get("channels", None)
        if isinstance(chs, (list, tuple)) and len(chs):
            args.metrics_channels = list(map(str, chs))

    if getattr(args, "metrics_conditions", None) is None:
        conds = erp_cfg.get("conditions", None)
        if conds is None:
            conds = metrics_cfg.get("conditions", None)
        if isinstance(conds, (list, tuple)) and len(conds):
            args.metrics_conditions = list(map(str, conds))
        elif conds is not None:
            args.metrics_conditions = [str(conds)]

    if args.erp_window is None:
        wins = erp_cfg.get("windows", None)
        if wins is None:
            wins = metrics_cfg.get("erp_windows", None)
        if isinstance(wins, list) and len(wins):
            parsed = []
            for window in wins:
                if isinstance(window, dict):
                    parsed.append([str(window.get("name", "window")), float(window.get("tmin")), float(window.get("tmax"))])
                elif isinstance(window, (list, tuple)) and len(window) >= 3:
                    parsed.append([str(window[0]), float(window[1]), float(window[2])])
            if parsed:
                args.erp_window = parsed

    set_if_default(
        args,
        defaults,
        "compute_mmn",
        int(bool(erp_cfg.get("compute_mmn", metrics_cfg.get("compute_mmn", args.compute_mmn)))),
    )
    set_if_default(
        args,
        defaults,
        "difference_label",
        erp_cfg.get("difference_label", metrics_cfg.get("difference_label", args.difference_label)),
    )
    set_if_default(
        args,
        defaults,
        "compute_p300",
        int(bool(erp_cfg.get("compute_p300", metrics_cfg.get("compute_p300", args.compute_p300)))),
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
    baseline = tfr_cfg.get("baseline", [args.tfr_baseline[0], args.tfr_baseline[1]])
    if isinstance(baseline, (list, tuple)) and len(baseline) >= 2:
        set_if_default(args, defaults, "tfr_baseline", [float(baseline[0]), float(baseline[1])])
    set_if_default(
        args,
        defaults,
        "tfr_baseline_mode",
        tfr_cfg.get("baseline_mode", tfr_cfg.get("mode", args.tfr_baseline_mode)),
    )

    compute_cfg = cfg.get("compute", {})
    set_if_default(args, defaults, "use_gpu", bool(compute_cfg.get("use_gpu", args.use_gpu)))
    set_if_default(args, defaults, "gpu_device", compute_cfg.get("gpu_device", args.gpu_device))

    if args.token_map is None:
        token_map = cfg.get("labels", {}).get("token_map", None)
        if isinstance(token_map, dict):
            args.token_map = [f"{key}={value}" for key, value in token_map.items()]
        elif isinstance(token_map, list):
            args.token_map = token_map
        else:
            args.token_map = None

    return cfg


def apply_erp_core_preset(args: Namespace, defaults: dict) -> None:
    """Apply ERP CORE-style defaults before config values are merged."""
    if not getattr(args, "erp_core", False):
        return
    args._erp_core_preset_enabled = True
    set_if_default(args, defaults, "reref", "tp9_tp10")
    set_if_default(args, defaults, "l_freq", 0.1)
    set_if_default(args, defaults, "h_freq", 20.0)
    set_if_default(args, defaults, "volt_method", "simple")
    set_if_default(args, defaults, "volt_auto_percentile", 97.5)
    set_if_default(args, defaults, "blink_auto_percentile", 99.0)
    set_if_default(args, defaults, "ica", "on")
