# eeg_pipeline/config.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


# ----------------------------
# Public API
# ----------------------------
def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a config file (YAML preferred; JSON supported) and validate/normalize it.

    YAML requires PyYAML:
      pip install pyyaml

    Returns a plain dict with defaults filled and types normalized.
    Raises ValueError with a readable message on validation failure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = _read_config_file(path)
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping/dict.")

    cfg = _apply_defaults(cfg)
    cfg = _normalize_config(cfg)   # normalize types (including paths -> Path)
    _validate_config(cfg)

    return cfg


def config_get(cfg: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Simple dotted-path getter: config_get(cfg, 'events.standard_codes')."""
    cur: Any = cfg
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ----------------------------
# Reading
# ----------------------------
def _read_config_file(path: Path) -> Dict[str, Any]:
    suffix = path.suffix.lower()

    if suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise ImportError(
                "YAML config requires PyYAML. Install with: pip install pyyaml"
            ) from e
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config extension '{suffix}'. Use .yml/.yaml or .json")


# ----------------------------
# Defaults (minimal + sane)
# ----------------------------
def _apply_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Only set defaults where missing; do not overwrite user values.
    def set_default(d: Dict[str, Any], path: str, value: Any):
        parts = path.split(".")
        cur = d
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur.setdefault(parts[-1], value)

    set_default(cfg, "task", "unknown")

    set_default(cfg, "paths.raw_dir", None)
    set_default(cfg, "paths.subject_csv_dir", None)
    set_default(cfg, "paths.out_dir", None)

    set_default(cfg, "channels.picks", [])
    set_default(cfg, "channels.eog_chs", [])
    set_default(cfg, "channels.blink_proxy_chs", ["Fp1"])
    set_default(cfg, "channels.drop_aux_chs", ["AUX"])

    set_default(cfg, "preprocess.montage", "standard_1020")
    set_default(cfg, "preprocess.reref", "average")  # average | none | p9_p10/tp9_tp10
    set_default(cfg, "preprocess.notch_hz", [60.0])
    set_default(cfg, "preprocess.l_freq", 0.1)
    set_default(cfg, "preprocess.h_freq", 30.0)
    set_default(cfg, "preprocess.resample_hz", None)
    set_default(cfg, "preprocess.lowpass_after_asr", False)

    set_default(cfg, "events.behavioral_keep_codes", [])
    set_default(cfg, "events.standard_codes", [])
    set_default(cfg, "events.deviant_codes", [])
    set_default(cfg, "events.condition_map", None)
    set_default(cfg, "events.drop_eeg_markers_by_gap_s", None)
    set_default(cfg, "events.auto_drop_to_count", True)
    set_default(cfg, "events.shift_s", 0.0)

    set_default(cfg, "epoching.tmin", -0.2)
    set_default(cfg, "epoching.tmax", 0.6)
    set_default(cfg, "epoching.baseline", [-0.2, 0.0])

    set_default(cfg, "artifacts.test_window", [-0.2, 0.3])
    set_default(cfg, "artifacts.max_reject_rate", None)
    set_default(cfg, "artifacts.blink.threshold_uv", 75.0)
    set_default(cfg, "artifacts.blink.win_ms", 200.0)
    set_default(cfg, "artifacts.blink.step_ms", 10.0)
    set_default(cfg, "artifacts.blink.auto_percentile", None)
    set_default(cfg, "artifacts.voltage.method", "simple")  # simple | window_ptp
    set_default(cfg, "artifacts.voltage.threshold_uv", 150.0)
    set_default(cfg, "artifacts.voltage.win_ms", 200.0)
    set_default(cfg, "artifacts.voltage.step_ms", 10.0)
    set_default(cfg, "artifacts.voltage.pos_uv", 150.0)
    set_default(cfg, "artifacts.voltage.neg_uv", -150.0)
    set_default(cfg, "artifacts.voltage.step_uv_per_ms", None)
    set_default(cfg, "artifacts.voltage.auto_percentile", None)

    set_default(cfg, "asr.enabled", False)
    set_default(cfg, "asr.cutoff", 20.0)
    set_default(cfg, "asr.blocksize", 100)
    set_default(cfg, "asr.win_len", 0.5)
    set_default(cfg, "asr.win_overlap", 0.66)
    set_default(cfg, "asr.max_dropout_fraction", 0.1)
    set_default(cfg, "asr.min_clean_fraction", 0.25)
    set_default(cfg, "asr.max_bad_chans", 0.1)
    set_default(cfg, "asr.method", "euclid")
    set_default(cfg, "asr.lookahead", 0.25)
    set_default(cfg, "asr.stepsize", 32)
    set_default(cfg, "asr.maxdims", 0.66)

    set_default(cfg, "ica.mode", "off")  # off | auto | on
    set_default(cfg, "ica.auto_blink_rate_per_min", 15.0)
    set_default(cfg, "ica.method", "fastica")
    set_default(cfg, "ica.n_components", 0.99)  # float variance fraction OR int #components (we normalize below)
    set_default(cfg, "ica.random_state", 97)
    set_default(cfg, "ica.max_iter", 512)
    set_default(cfg, "ica.fit_l_freq", 1.0)
    set_default(cfg, "ica.fit_h_freq", None)
    set_default(cfg, "ica.decim", 3)
    set_default(cfg, "ica.corr_thresh", 0.30)
    set_default(cfg, "ica.max_exclude", 3)
    set_default(cfg, "ica.save_ica", True)
    set_default(cfg, "ica.infomax_extended", True)
    set_default(cfg, "ica.iclabel.enabled", False)
    set_default(cfg, "ica.iclabel.threshold", 0.9)
    set_default(cfg, "ica.iclabel.classes", ["ocular", "cardiac"])

    set_default(cfg, "labels.token_map", None)

    set_default(cfg, "compute.use_gpu", False)
    set_default(cfg, "compute.gpu_device", None)

    set_default(cfg, "metrics.erp.enabled", True)
    set_default(cfg, "metrics.erp.windows", [])
    set_default(cfg, "metrics.erp.timeseries", False)
    set_default(cfg, "metrics.erp.difference_label", None)
    set_default(cfg, "metrics.erp.conditions", None)
    set_default(cfg, "metrics.erp.grand_average_weighting", "equal")

    set_default(cfg, "metrics.tfr.enabled", False)
    set_default(cfg, "metrics.tfr.method", "multitaper")
    set_default(cfg, "metrics.tfr.tmin", -0.2)
    set_default(cfg, "metrics.tfr.tmax", 0.6)
    set_default(cfg, "metrics.tfr.fmin", 1.0)
    set_default(cfg, "metrics.tfr.fmax", 30.0)
    set_default(cfg, "metrics.tfr.baseline", [-0.2, 0.0])
    set_default(cfg, "metrics.tfr.baseline_mode", "logratio")

    return cfg


# ----------------------------
# Validation
# ----------------------------
def _validate_config(cfg: Dict[str, Any]) -> None:
    errors: List[str] = []

    def require(path: str):
        val = config_get(cfg, path, None)
        if val is None or (isinstance(val, str) and not val.strip()):
            errors.append(f"Missing required field: '{path}'")

    # Paths required for the pipeline proper (you can relax for “metrics-only” runs later)
    require("paths.raw_dir")
    require("paths.subject_csv_dir")
    require("paths.out_dir")

    # Events required if doing ERP contrasts
    std = config_get(cfg, "events.standard_codes", [])
    dev = config_get(cfg, "events.deviant_codes", [])
    if not std or not dev:
        errors.append("events.standard_codes and events.deviant_codes must both be non-empty.")

    # Baseline shape
    baseline = config_get(cfg, "epoching.baseline", None)
    if not (isinstance(baseline, list) and len(baseline) == 2):
        errors.append("epoching.baseline must be a 2-item list: [tmin, tmax].")

    # Preprocess sanity
    lf = config_get(cfg, "preprocess.l_freq", None)
    hf = config_get(cfg, "preprocess.h_freq", None)
    if lf not in (None, "null", "None"):
        try:
            if float(lf) < 0:
                errors.append("preprocess.l_freq must be >= 0 (or null).")
        except Exception:
            errors.append("preprocess.l_freq must be a number (or null).")
    if hf not in (None, "null", "None"):
        try:
            if float(hf) <= 0:
                errors.append("preprocess.h_freq must be > 0 (or null).")
        except Exception:
            errors.append("preprocess.h_freq must be a number (or null).")
    rs = config_get(cfg, "preprocess.resample_hz", None)
    if rs not in (None, "null", "None"):
        try:
            if float(rs) <= 0:
                errors.append("preprocess.resample_hz must be > 0 (or null).")
        except Exception:
            errors.append("preprocess.resample_hz must be a number (or null).")

    # ICA mode
    ica_mode = str(config_get(cfg, "ica.mode", "off")).lower()
    if ica_mode not in {"off", "auto", "on"}:
        errors.append("ica.mode must be one of: off | auto | on.")

    # Re-reference mode
    reref_mode = str(config_get(cfg, "preprocess.reref", "average")).lower()
    if reref_mode not in {"average", "avg", "none", "no", "p9_p10", "tp9_tp10", "mastoids", "mastoid", "linked_mastoids", "linked"}:
        errors.append("preprocess.reref must be one of: average | none | p9_p10/tp9_tp10 (mastoids).")

    volt_method = str(config_get(cfg, "artifacts.voltage.method", "simple")).lower()
    if volt_method not in {"simple", "window_ptp", "combined"}:
        errors.append("artifacts.voltage.method must be one of: simple | window_ptp | combined.")

    blink_auto = config_get(cfg, "artifacts.blink.auto_percentile", None)
    if blink_auto not in (None, "null", "None"):
        try:
            blink_auto = float(blink_auto)
            if not (0 < blink_auto <= 100):
                errors.append("artifacts.blink.auto_percentile must be in (0, 100].")
        except Exception:
            errors.append("artifacts.blink.auto_percentile must be a number in (0, 100].")

    volt_auto = config_get(cfg, "artifacts.voltage.auto_percentile", None)
    if volt_auto not in (None, "null", "None"):
        try:
            volt_auto = float(volt_auto)
            if not (0 < volt_auto <= 100):
                errors.append("artifacts.voltage.auto_percentile must be in (0, 100].")
        except Exception:
            errors.append("artifacts.voltage.auto_percentile must be a number in (0, 100].")

    # Grand-average weighting
    ga_w = str(config_get(cfg, "metrics.erp.grand_average_weighting", "equal")).lower()
    if ga_w not in {"equal", "nave"}:
        errors.append("metrics.erp.grand_average_weighting must be 'equal' or 'nave'.")

    # Disjoint standard/deviant
    try:
        std_set = set(int(x) for x in std)
        dev_set = set(int(x) for x in dev)
        overlap = sorted(std_set.intersection(dev_set))
        if overlap:
            errors.append(f"Standard/deviant code overlap not allowed: {overlap}")
    except Exception:
        errors.append("events.standard_codes / deviant_codes must be integers.")

    # Condition map (optional)
    cond_map = config_get(cfg, "events.condition_map", None)
    if cond_map is not None:
        if not isinstance(cond_map, dict):
            errors.append("events.condition_map must be a mapping of name -> code(s).")
        else:
            seen = set()
            try:
                for name, codes in cond_map.items():
                    codes_list = codes if isinstance(codes, (list, tuple, set)) else [codes]
                    if len(codes_list) != 1:
                        errors.append(
                            f"events.condition_map['{name}'] must map to a single code."
                        )
                        continue
                    c = int(list(codes_list)[0])
                    if c in seen:
                        errors.append(f"events.condition_map has duplicate code: {c}")
                    seen.add(c)
            except Exception:
                errors.append("events.condition_map values must be integers (or lists of integers).")

    # ERP windows sanity
    erp_windows = config_get(cfg, "metrics.erp.windows", [])
    if erp_windows:
        if not isinstance(erp_windows, list):
            errors.append("metrics.erp.windows must be a list.")
        else:
            for i, w in enumerate(erp_windows):
                if not isinstance(w, dict):
                    errors.append(f"metrics.erp.windows[{i}] must be a mapping.")
                    continue
                if "name" not in w or "tmin" not in w or "tmax" not in w:
                    errors.append(f"metrics.erp.windows[{i}] must include name, tmin, tmax.")

    # If keep_codes is set, enforce std/dev subset
    keep_codes = config_get(cfg, "events.behavioral_keep_codes", [])
    if keep_codes:
        try:
            keep_set = set(int(x) for x in keep_codes)
            if not set(int(x) for x in std).issubset(keep_set):
                errors.append(
                    "events.standard_codes must be included in events.behavioral_keep_codes (or set behavioral_keep_codes empty)."
                )
            if not set(int(x) for x in dev).issubset(keep_set):
                errors.append(
                    "events.deviant_codes must be included in events.behavioral_keep_codes (or set behavioral_keep_codes empty)."
                )
        except Exception:
            errors.append("events.behavioral_keep_codes must be integers.")

    if errors:
        msg = "Config validation failed:\n  - " + "\n  - ".join(errors)
        raise ValueError(msg)


# ----------------------------
# Normalization (types + conveniences)
# ----------------------------
def _normalize_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(cfg)  # shallow copy; mutate nested dicts

    # Paths -> Path objects (IMPORTANT: do NOT convert back to str)
    for k in ("raw_dir", "subject_csv_dir", "out_dir"):
        v = cfg["paths"].get(k)
        if v is not None:
            cfg["paths"][k] = Path(v)

    # Int lists
    cfg["events"]["behavioral_keep_codes"] = _as_int_list(cfg["events"].get("behavioral_keep_codes", []))
    cfg["events"]["standard_codes"] = _as_int_list(cfg["events"].get("standard_codes", []))
    cfg["events"]["deviant_codes"] = _as_int_list(cfg["events"].get("deviant_codes", []))
    cfg["events"]["condition_map"] = _normalize_condition_map(cfg["events"].get("condition_map", None))
    cfg["events"]["shift_s"] = float(cfg["events"].get("shift_s", 0.0))

    # Floats
    lf = cfg["preprocess"].get("l_freq", None)
    cfg["preprocess"]["l_freq"] = None if lf in (None, "null", "None") else float(lf)
    hf = cfg["preprocess"].get("h_freq", None)
    cfg["preprocess"]["h_freq"] = None if hf in (None, "null", "None") else float(hf)
    rs = cfg["preprocess"].get("resample_hz", None)
    cfg["preprocess"]["resample_hz"] = None if rs in (None, "null", "None") else float(rs)
    cfg["preprocess"]["lowpass_after_asr"] = bool(cfg["preprocess"].get("lowpass_after_asr", False))
    cfg["preprocess"]["notch_hz"] = _as_float_list(cfg["preprocess"].get("notch_hz", []))

    # Epoching
    cfg["epoching"]["tmin"] = float(cfg["epoching"]["tmin"])
    cfg["epoching"]["tmax"] = float(cfg["epoching"]["tmax"])
    cfg["epoching"]["baseline"] = [
        float(cfg["epoching"]["baseline"][0]),
        float(cfg["epoching"]["baseline"][1]),
    ]

    # Artifact windows
    cfg["artifacts"]["test_window"] = [
        float(cfg["artifacts"]["test_window"][0]),
        float(cfg["artifacts"]["test_window"][1]),
    ]
    cfg["artifacts"]["max_reject_rate"] = (
        None if cfg["artifacts"]["max_reject_rate"] in (None, "null", "None") else float(cfg["artifacts"]["max_reject_rate"])
    )
    cfg["artifacts"]["voltage"]["method"] = str(cfg["artifacts"]["voltage"].get("method", "simple")).lower()
    cfg["artifacts"]["voltage"]["threshold_uv"] = float(cfg["artifacts"]["voltage"].get("threshold_uv", 150.0))
    cfg["artifacts"]["voltage"]["win_ms"] = float(cfg["artifacts"]["voltage"].get("win_ms", 200.0))
    cfg["artifacts"]["voltage"]["step_ms"] = float(cfg["artifacts"]["voltage"].get("step_ms", 10.0))
    step_uv = cfg["artifacts"]["voltage"].get("step_uv_per_ms", None)
    cfg["artifacts"]["voltage"]["step_uv_per_ms"] = (
        None if step_uv in (None, "null", "None") else float(step_uv)
    )
    blink_auto = cfg["artifacts"]["blink"].get("auto_percentile", None)
    cfg["artifacts"]["blink"]["auto_percentile"] = (
        None if blink_auto in (None, "null", "None") else float(blink_auto)
    )
    volt_auto = cfg["artifacts"]["voltage"].get("auto_percentile", None)
    cfg["artifacts"]["voltage"]["auto_percentile"] = (
        None if volt_auto in (None, "null", "None") else float(volt_auto)
    )

    # ASR
    asr_cfg = cfg.get("asr", {})
    cfg["asr"]["enabled"] = bool(asr_cfg.get("enabled", False))
    cfg["asr"]["cutoff"] = float(asr_cfg.get("cutoff", 20.0))
    cfg["asr"]["blocksize"] = int(asr_cfg.get("blocksize", 100))
    cfg["asr"]["win_len"] = float(asr_cfg.get("win_len", 0.5))
    cfg["asr"]["win_overlap"] = float(asr_cfg.get("win_overlap", 0.66))
    cfg["asr"]["max_dropout_fraction"] = float(asr_cfg.get("max_dropout_fraction", 0.1))
    cfg["asr"]["min_clean_fraction"] = float(asr_cfg.get("min_clean_fraction", 0.25))
    cfg["asr"]["max_bad_chans"] = float(asr_cfg.get("max_bad_chans", 0.1))
    cfg["asr"]["method"] = str(asr_cfg.get("method", "euclid"))
    cfg["asr"]["lookahead"] = float(asr_cfg.get("lookahead", 0.25))
    cfg["asr"]["stepsize"] = int(asr_cfg.get("stepsize", 32))
    cfg["asr"]["maxdims"] = float(asr_cfg.get("maxdims", 0.66))

    # ICA
    cfg["ica"]["mode"] = str(cfg["ica"]["mode"]).lower()
    cfg["ica"]["auto_blink_rate_per_min"] = float(cfg["ica"]["auto_blink_rate_per_min"])
    cfg["ica"]["n_components"] = _parse_n_components(cfg["ica"]["n_components"])
    cfg["ica"]["random_state"] = int(cfg["ica"]["random_state"])
    cfg["ica"]["max_iter"] = int(cfg["ica"]["max_iter"])
    cfg["ica"]["fit_l_freq"] = float(cfg["ica"]["fit_l_freq"])
    cfg["ica"]["fit_h_freq"] = None if cfg["ica"]["fit_h_freq"] in (None, "null", "None") else float(cfg["ica"]["fit_h_freq"])
    cfg["ica"]["decim"] = int(cfg["ica"]["decim"])
    cfg["ica"]["corr_thresh"] = float(cfg["ica"]["corr_thresh"])
    cfg["ica"]["max_exclude"] = int(cfg["ica"]["max_exclude"])
    cfg["ica"]["save_ica"] = bool(cfg["ica"]["save_ica"])
    cfg["ica"]["infomax_extended"] = bool(cfg["ica"].get("infomax_extended", True))
    iclabel_cfg = cfg["ica"].get("iclabel", {})
    iclabel_classes = iclabel_cfg.get("classes", [])
    if isinstance(iclabel_classes, (list, tuple)):
        iclabel_classes = [str(c).lower() for c in iclabel_classes]
    else:
        iclabel_classes = [str(iclabel_classes).lower()]
    cfg["ica"]["iclabel"] = {
        "enabled": bool(iclabel_cfg.get("enabled", False)),
        "threshold": float(iclabel_cfg.get("threshold", 0.9)),
        "classes": iclabel_classes,
    }

    # Token map convenience normalization -> {"token1": "...", "token2": "..."} or None
    cfg["labels"]["token_map"] = _normalize_token_map(cfg["labels"].get("token_map"))

    # Metrics conditions (optional)
    conds = cfg["metrics"]["erp"].get("conditions", None)
    if conds is not None:
        if isinstance(conds, (list, tuple)):
            cfg["metrics"]["erp"]["conditions"] = [str(c) for c in conds]
        else:
            cfg["metrics"]["erp"]["conditions"] = [str(conds)]
    cfg["metrics"]["erp"]["grand_average_weighting"] = str(
        cfg["metrics"]["erp"].get("grand_average_weighting", "equal")
    ).lower()

    # Compute (optional GPU acceleration)
    compute_cfg = cfg.get("compute", {})
    cfg["compute"]["use_gpu"] = bool(compute_cfg.get("use_gpu", False))
    gd = compute_cfg.get("gpu_device", None)
    cfg["compute"]["gpu_device"] = None if gd in (None, "null", "None", "") else int(gd)

    return cfg


def _as_int_list(x: Any) -> List[int]:
    if x is None:
        return []
    if isinstance(x, (tuple, list)):
        return [int(v) for v in x]
    return [int(x)]


def _as_float_list(x: Any) -> List[float]:
    if x is None:
        return []
    if isinstance(x, (tuple, list)):
        return [float(v) for v in x]
    return [float(x)]


def _parse_n_components(x: Any) -> Union[int, float]:
    """
    MNE ICA n_components can be float (variance fraction) or int (#components).
    YAML may give int/float already; JSON may give number; CLI might give string.
    """
    if x is None:
        return 0.99
    if isinstance(x, (int, float)):
        return x
    s = str(x).strip()
    if not s:
        return 0.99
    # Accept "20" or "0.99"
    try:
        if any(ch in s for ch in (".", "e", "E")):
            return float(s)
        return int(s)
    except Exception:
        return float(s)


def _normalize_token_map(token_map: Any) -> Optional[Dict[str, str]]:
    if token_map is None:
        return None

    # Allow dict form in YAML
    if isinstance(token_map, dict):
        out: Dict[str, str] = {}
        for k, v in token_map.items():
            lk = str(k).strip().lower()
            if lk in {"token1", "token2"}:
                out[lk] = str(v)
        return out or None

    # Allow list form
    if isinstance(token_map, (list, tuple)):
        parts = [str(p) for p in token_map if str(p).strip()]

        # Shorthand: ["EH", "IH"]
        if len(parts) == 2 and all("=" not in p for p in parts):
            return {"token1": parts[0], "token2": parts[1]}

        # Keyed: ["Token1=EH", "Token2=IH"] (case-insensitive keys)
        out: Dict[str, str] = {}
        for p in parts:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            lk = k.strip().lower()
            if lk in {"token1", "token2"}:
                out[lk] = v.strip()
        return out or None

    # Allow single string (rare)
    s = str(token_map).strip()
    if not s:
        return None
    if " " in s and "=" not in s:
        a, b = s.split(None, 1)
        return {"token1": a, "token2": b}
    return None


def _normalize_condition_map(condition_map: Any) -> Optional[Dict[str, list[int]]]:
    if condition_map is None:
        return None
    if not isinstance(condition_map, dict):
        raise ValueError("events.condition_map must be a mapping of name -> code(s).")
    out: Dict[str, list[int]] = {}
    for k, v in condition_map.items():
        name = str(k)
        if isinstance(v, (list, tuple, set)):
            codes = [int(x) for x in v]
        else:
            codes = [int(v)]
        out[name] = codes
    return out
