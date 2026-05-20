#!/usr/bin/env python3
"""Import manually artifact-rejected EEGLAB epochs into pipeline format.

This utility supports a manual-rejection workflow:
1) Read epoched EEGLAB ``.set`` files (already cleaned/rejected by hand).
2) Recode condition labels to ``Standard`` / ``Deviant`` from config code lists.
3) Save pipeline-compatible FIF epochs to ``<out_dir>/02_epochs/*-epo.fif``.
4) Optionally run pipeline metrics (``--get_metrics``).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Avoid numba cache locator issues in some sandboxed/readonly environments.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

_PAREN_CODE_RE = re.compile(r"\((\d+)\)")
_TRAILING_CODE_RE = re.compile(r"(\d+)\s*$")
_REJECT_FIELDS = {
    "manual": ("rejmanual",),
    "global": ("rejglobal",),
    "manual_or_global": ("rejmanual", "rejglobal"),
    "all": ("rejmanual", "rejglobal", "rejconst", "rejfreq", "rejjp", "rejkurt", "rejthresh"),
    "none": (),
}


def _read_config_file(path: Path) -> dict:
    suf = path.suffix.lower()
    if suf in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise ImportError("YAML config requires PyYAML.") from e
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data or {}
    if suf == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    raise ValueError(f"Unsupported config extension: {suf}")


def _load_minimal_config(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    cfg = _read_config_file(p)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping/dict.")
    if "events" not in cfg or not isinstance(cfg["events"], dict):
        raise ValueError("Config must include events.standard_codes and events.deviant_codes.")
    if "paths" not in cfg or not isinstance(cfg["paths"], dict):
        cfg["paths"] = {}
    return cfg


def _event_code_from_label(label: str) -> int | None:
    m = _PAREN_CODE_RE.search(label)
    if m:
        return int(m.group(1))
    m = _TRAILING_CODE_RE.search(label)
    if m:
        return int(m.group(1))
    return None


@dataclass
class ConvertRow:
    subject: str
    source_set: str
    output_fif: str
    n_total_epochs: int
    n_flagged_rejected: int
    n_standard: int
    n_deviant: int
    reject_mode: str
    reject_details: str
    status: str
    error: str


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Import manually artifact-rejected EEGLAB .set epochs and optionally run metrics."
    )
    ap.add_argument("--config", required=True, help="Path to pipeline YAML/JSON config.")
    ap.add_argument("--manual_set_dir", required=True, help="Folder containing epoched .set files.")
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Pipeline output root (default: paths.out_dir from config).",
    )
    ap.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Optional subject stems to include (e.g., s201 s202 s205-2).",
    )
    ap.add_argument(
        "--run_metrics",
        action="store_true",
        help="If set, run pipeline --get_metrics after import.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing *-epo.fif files.",
    )
    ap.add_argument(
        "--reject_mode",
        choices=tuple(_REJECT_FIELDS.keys()),
        default="manual_or_global",
        help=(
            "Which EEGLAB trial-rejection flags to honor before saving FIF. "
            "Default: manual_or_global."
        ),
    )
    return ap.parse_args(argv)


def _iter_set_files(root: Path, subjects: Iterable[str] | None) -> list[Path]:
    files = sorted(p for p in root.rglob("*.set") if p.is_file())
    if not subjects:
        return files
    wanted = {s.lower() for s in subjects}
    return [p for p in files if p.stem.lower() in wanted]


def _field_get(obj: object, name: str) -> object | None:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _load_reject_struct(set_path: Path) -> object | None:
    mat = loadmat(set_path, struct_as_record=False, squeeze_me=True)
    reject = _field_get(mat, "reject")
    if reject is not None:
        return reject
    eeg = _field_get(mat, "EEG")
    if eeg is not None:
        return _field_get(eeg, "reject")
    return None


def _reject_mask_from_set(set_path: Path, n_trials: int, reject_mode: str) -> tuple[np.ndarray | None, dict[str, int]]:
    if reject_mode == "none":
        return None, {}

    reject = _load_reject_struct(set_path)
    if reject is None:
        return None, {}

    masks: dict[str, np.ndarray] = {}
    for field in _REJECT_FIELDS[reject_mode]:
        raw = _field_get(reject, field)
        if raw is None:
            continue
        arr = np.asarray(raw).squeeze()
        if arr.size == 0:
            continue
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        if arr.size != n_trials:
            raise ValueError(
                f"{set_path.name}: reject.{field} has length {arr.size}, expected {n_trials} trials"
            )
        masks[field] = arr.astype(float) > 0

    if not masks:
        return None, {}

    combined = np.logical_or.reduce(list(masks.values()))
    stats = {f"flagged_{name}": int(mask.sum()) for name, mask in masks.items()}
    stats["flagged_total"] = int(combined.sum())
    return combined, stats


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = _load_minimal_config(args.config)

    std_codes = set(int(x) for x in cfg["events"].get("standard_codes", []))
    dev_codes = set(int(x) for x in cfg["events"].get("deviant_codes", []))
    if not std_codes or not dev_codes:
        raise ValueError("Config must define non-empty events.standard_codes and events.deviant_codes.")

    manual_set_dir = Path(args.manual_set_dir)
    if not manual_set_dir.exists():
        raise FileNotFoundError(f"manual_set_dir not found: {manual_set_dir}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        cfg_out = cfg["paths"].get("out_dir", None)
        if not cfg_out:
            raise ValueError("Config paths.out_dir is missing; pass --out_dir explicitly.")
        out_dir = Path(cfg_out)
    epochs_dir = out_dir / "02_epochs"
    epochs_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_set_files(manual_set_dir, args.subjects)
    if not files:
        raise RuntimeError(f"No .set files found in {manual_set_dir}")

    rows: list[ConvertRow] = []
    for set_path in files:
        subj = set_path.stem
        out_fif = epochs_dir / f"{subj}-epo.fif"

        try:
            if out_fif.exists() and not args.overwrite:
                rows.append(
                    ConvertRow(
                        subject=subj,
                        source_set=str(set_path),
                        output_fif=str(out_fif),
                        n_total_epochs=0,
                        n_flagged_rejected=0,
                        n_standard=0,
                        n_deviant=0,
                        reject_mode=args.reject_mode,
                        reject_details="{}",
                        status="SKIP_EXISTS",
                        error=f"Exists (use --overwrite): {out_fif}",
                    )
                )
                continue

            epochs = mne.io.read_epochs_eeglab(set_path, verbose="error")
            n_total_epochs = int(len(epochs))
            reject_mask, reject_stats = _reject_mask_from_set(set_path, n_total_epochs, args.reject_mode)
            n_flagged_rejected = int(reject_stats.get("flagged_total", 0))
            if reject_mask is not None:
                keep_idx = np.flatnonzero(~reject_mask)
                epochs = epochs[keep_idx].copy()

            std_labels = [k for k in epochs.event_id if _event_code_from_label(k) in std_codes]
            dev_labels = [k for k in epochs.event_id if _event_code_from_label(k) in dev_codes]
            keep = std_labels + dev_labels
            if not std_labels or not dev_labels:
                raise ValueError(
                    f"Could not map Standard/Deviant labels from event_id keys: {sorted(epochs.event_id.keys())}"
                )

            epochs = epochs[keep].copy()
            mne.epochs.combine_event_ids(epochs, std_labels, {"Standard": int(sorted(std_codes)[0])}, copy=False)
            mne.epochs.combine_event_ids(epochs, dev_labels, {"Deviant": int(sorted(dev_codes)[0])}, copy=False)
            epochs.save(out_fif, overwrite=True, verbose="error")

            rows.append(
                ConvertRow(
                    subject=subj,
                    source_set=str(set_path),
                    output_fif=str(out_fif),
                    n_total_epochs=n_total_epochs,
                    n_flagged_rejected=n_flagged_rejected,
                    n_standard=int(len(epochs["Standard"])),
                    n_deviant=int(len(epochs["Deviant"])),
                    reject_mode=args.reject_mode,
                    reject_details=json.dumps(reject_stats, sort_keys=True),
                    status="OK",
                    error="",
                )
            )
            print(
                f"[OK] {subj}: total={n_total_epochs} flagged={n_flagged_rejected} "
                f"Standard={len(epochs['Standard'])} Deviant={len(epochs['Deviant'])}"
            )

        except Exception as e:  # pragma: no cover - utility script
            rows.append(
                ConvertRow(
                    subject=subj,
                    source_set=str(set_path),
                    output_fif=str(out_fif),
                    n_total_epochs=0,
                    n_flagged_rejected=0,
                    n_standard=0,
                    n_deviant=0,
                    reject_mode=args.reject_mode,
                    reject_details="{}",
                    status="ERROR",
                    error=str(e),
                )
            )
            print(f"[ERROR] {subj}: {e}")

    summary = pd.DataFrame([r.__dict__ for r in rows])
    summary_path = out_dir / "manual_rejection_import_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved import summary -> {summary_path}")
    print(summary["status"].value_counts().to_string())

    if args.run_metrics:
        print("\nRunning pipeline metrics...")
        cmd = [os.environ.get("PYTHON", "python3"), "-m", "eeg_pipeline.cli", "--config", str(args.config), "--out_dir", str(out_dir), "--get_metrics"]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
