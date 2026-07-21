from __future__ import annotations

from .cli_common import _finalize_runtime_paths, load_config, set_if_default


def _set_override(overrides: dict, path: tuple[str, ...], value) -> None:
    cur = overrides
    for part in path[:-1]:
        cur = cur.setdefault(part, {})
    cur[path[-1]] = value


def _build_config_overrides(args, defaults: dict | None) -> dict:
    defaults = defaults or {}
    overrides: dict = {}

    def provided(field: str) -> bool:
        if field not in defaults:
            return False
        return getattr(args, field) != defaults[field]

    scalar_paths = {
        "bids_root": ("paths", "bids_root"),
        "raw_dir": ("paths", "raw_dir"),
        "subject_csv_dir": ("paths", "subject_csv_dir"),
        "derivatives_root": ("paths", "derivatives_root"),
        "sourcedata_root": ("paths", "sourcedata_root"),
        "task_label": ("task",),
        "behavior_csv_fallback_dir": ("events", "csv_fallback_dir"),
        "montage": ("preprocess", "montage"),
        "reref": ("preprocess", "reref"),
        "l_freq": ("preprocess", "l_freq"),
        "h_freq": ("preprocess", "h_freq"),
        "notch": ("preprocess", "notch_hz"),
        "eog_chs": ("channels", "eog_chs"),
        "blink_proxy_chs": ("channels", "blink_proxy_chs"),
        "aux_chs": ("channels", "drop_aux_chs"),
        "standard_codes": ("events", "standard_codes"),
        "deviant_codes": ("events", "deviant_codes"),
        "behavioral_keep_codes": ("events", "behavioral_keep_codes"),
        "drop_eeg_markers_by_gap_s": ("events", "drop_eeg_markers_by_gap_s"),
        "auto_drop_to_count": ("events", "auto_drop_to_count"),
        "blink_threshold_uv": ("artifacts", "blink", "threshold_uv"),
        "blink_win_ms": ("artifacts", "blink", "win_ms"),
        "blink_step_ms": ("artifacts", "blink", "step_ms"),
        "blink_auto_percentile": ("artifacts", "blink", "auto_percentile"),
        "volt_pos_uv": ("artifacts", "voltage", "pos_uv"),
        "volt_neg_uv": ("artifacts", "voltage", "neg_uv"),
        "volt_method": ("artifacts", "voltage", "method"),
        "volt_threshold_uv": ("artifacts", "voltage", "threshold_uv"),
        "volt_win_ms": ("artifacts", "voltage", "win_ms"),
        "volt_step_ms": ("artifacts", "voltage", "step_ms"),
        "volt_step_uv_per_ms": ("artifacts", "voltage", "step_uv_per_ms"),
        "volt_auto_percentile": ("artifacts", "voltage", "auto_percentile"),
        "max_reject_rate": ("artifacts", "max_reject_rate"),
        "ica": ("ica", "mode"),
        "ica_auto_blink_rate_per_min": ("ica", "auto_blink_rate_per_min"),
        "ica_method": ("ica", "method"),
        "ica_n_components": ("ica", "n_components"),
        "ica_random_state": ("ica", "random_state"),
        "ica_max_iter": ("ica", "max_iter"),
        "ica_fit_l_freq": ("ica", "fit_l_freq"),
        "ica_fit_h_freq": ("ica", "fit_h_freq"),
        "ica_decim": ("ica", "decim"),
        "ica_corr_thresh": ("ica", "corr_thresh"),
        "ica_max_exclude": ("ica", "max_exclude"),
        "save_ica": ("ica", "save_ica"),
        "token_map": ("labels", "token_map"),
        "use_gpu": ("compute", "use_gpu"),
        "gpu_device": ("compute", "gpu_device"),
        "convert_to_bids": ("conversion", "enabled"),
        "conversion_bids_root": ("conversion", "bids_output_root"),
        "conversion_overwrite": ("conversion", "overwrite"),
    }
    list_paths = {
        "subjects": ("bids", "subjects"),
        "sessions": ("bids", "sessions"),
        "tasks": ("bids", "tasks"),
        "runs": ("bids", "runs"),
    }

    if getattr(args, "legacy", False):
        _set_override(overrides, ("input", "mode"), "legacy")

    for field, path in scalar_paths.items():
        if provided(field):
            _set_override(overrides, path, getattr(args, field))

    for field, path in list_paths.items():
        if provided(field):
            _set_override(overrides, path, list(getattr(args, field)))

    if provided("tmin"):
        _set_override(overrides, ("epoching", "tmin"), args.tmin)
    if provided("tmax"):
        _set_override(overrides, ("epoching", "tmax"), args.tmax)
    if provided("baseline"):
        _set_override(overrides, ("epoching", "baseline"), list(args.baseline))
    if provided("art_test_tmin") or provided("art_test_tmax"):
        _set_override(overrides, ("artifacts", "test_window"), [args.art_test_tmin, args.art_test_tmax])

    return overrides


def apply_config(args, defaults=None):
    """Load config and apply values to args (respecting CLI overrides)."""
    if defaults is None:
        defaults = {}
    config_overrides = _build_config_overrides(args, defaults)
    if config_overrides:
        cfg = load_config(args.config, overrides=config_overrides)
    else:
        cfg = load_config(args.config)

    # Paths
    set_if_default(args, defaults, "raw_dir", cfg["paths"].get("raw_dir"))
    set_if_default(args, defaults, "subject_csv_dir", cfg["paths"].get("subject_csv_dir"))
    set_if_default(args, defaults, "bids_root", cfg["paths"]["bids_root"])
    set_if_default(args, defaults, "derivatives_root", cfg["paths"]["derivatives_root"])
    set_if_default(args, defaults, "sourcedata_root", cfg["paths"].get("sourcedata_root"))
    set_if_default(args, defaults, "conversion_bids_root", cfg.get("conversion", {}).get("bids_output_root"))
    set_if_default(
        args,
        defaults,
        "conversion_overwrite",
        int(bool(cfg.get("conversion", {}).get("overwrite", getattr(args, "conversion_overwrite", 1)))),
    )
    set_if_default(
        args,
        defaults,
        "convert_to_bids",
        bool(cfg.get("conversion", {}).get("enabled", getattr(args, "convert_to_bids", False))),
    )
    set_if_default(args, defaults, "task_label", cfg.get("task", getattr(args, "task_label", None)))
    if getattr(args, "legacy", False):
        args.input_mode = "legacy"
    else:
        args.input_mode = str(cfg.get("input", {}).get("mode", "bids")).lower()

    # BIDS discovery filters
    bids_cfg = cfg.get("bids", {})
    if getattr(args, "subjects", None) is None:
        args.subjects = bids_cfg.get("subjects", None)
    if getattr(args, "sessions", None) is None:
        args.sessions = bids_cfg.get("sessions", None)
    if getattr(args, "tasks", None) is None:
        args.tasks = bids_cfg.get("tasks", None)
    if getattr(args, "runs", None) is None:
        args.runs = bids_cfg.get("runs", None)

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
        args, defaults, "behavior_csv_fallback_dir",
        cfg["events"].get("csv_fallback_dir", getattr(args, "behavior_csv_fallback_dir", None)),
    )
    set_if_default(
        args, defaults, "auto_drop_to_count",
        int(bool(cfg["events"].get("auto_drop_to_count", args.auto_drop_to_count)))
    )
    # Optional condition map (name -> code)
    cond_map = cfg["events"].get("condition_map", None)
    if cond_map is not None:
        args.condition_map = cond_map

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
    set_if_default(args, defaults, "blink_auto_percentile", blink_cfg.get("auto_percentile", args.blink_auto_percentile))
    volt_cfg = art.get("voltage", {})
    set_if_default(args, defaults, "volt_pos_uv", volt_cfg.get("pos_uv", args.volt_pos_uv))
    set_if_default(args, defaults, "volt_neg_uv", volt_cfg.get("neg_uv", args.volt_neg_uv))
    # Optional windowed EEG artifact rejection (if configured)
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

    # Optional metrics conditions (for ERP/TFR)
    if getattr(args, "metrics_conditions", None) is None:
        conds = erp_cfg.get("conditions", None)
        if conds is None:
            conds = metrics_cfg.get("conditions", None)
        if isinstance(conds, (list, tuple)) and len(conds):
            args.metrics_conditions = list(map(str, conds))
        elif conds is not None:
            args.metrics_conditions = [str(conds)]

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
    b = tfr_cfg.get("baseline", [args.tfr_baseline[0], args.tfr_baseline[1]])
    if isinstance(b, (list, tuple)) and len(b) >= 2:
        set_if_default(args, defaults, "tfr_baseline", [float(b[0]), float(b[1])])
    set_if_default(
        args,
        defaults,
        "tfr_baseline_mode",
        tfr_cfg.get("baseline_mode", tfr_cfg.get("mode", args.tfr_baseline_mode)),
    )

    # Compute
    compute_cfg = cfg.get("compute", {})
    set_if_default(args, defaults, "use_gpu", bool(compute_cfg.get("use_gpu", args.use_gpu)))
    set_if_default(args, defaults, "gpu_device", compute_cfg.get("gpu_device", args.gpu_device))

    # Token map
    if args.token_map is None:
        tm = cfg.get("labels", {}).get("token_map", None)
        if isinstance(tm, dict):
            args.token_map = [f"{k}={v}" for k, v in tm.items()]
        elif isinstance(tm, list):
            args.token_map = tm
        else:
            args.token_map = None

    _finalize_runtime_paths(args, cfg)
    return cfg


def apply_erp_core_preset(args, defaults):
    """Apply ERP CORE-style defaults (TP9/TP10, 0.1–20 Hz, ICA on, individualized thresholds)."""
    if not getattr(args, "erp_core", False):
        return
    # Store for logging
    args._erp_core_preset_enabled = True
    # Preprocessing: ERP CORE uses TP9/TP10 and 0.1 Hz high-pass.
    set_if_default(args, defaults, "reref", "tp9_tp10")
    set_if_default(args, defaults, "l_freq", 0.1)
    # Apply 20 Hz low-pass to align with ERP CORE measurement filtering.
    set_if_default(args, defaults, "h_freq", 20.0)
    # Artifact thresholds: individualized via percentile-based rule.
    set_if_default(args, defaults, "volt_method", "simple")
    set_if_default(args, defaults, "volt_auto_percentile", 97.5)
    set_if_default(args, defaults, "blink_auto_percentile", 99.0)
    # ERP CORE runs ICA by default.
    set_if_default(args, defaults, "ica", "on")

