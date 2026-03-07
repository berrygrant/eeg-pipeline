from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

import mne

SIDECAR_SCHEMA_VERSION = 1

ReviewMode = Literal["auto", "raw", "epochs"]
ResolvedReviewMode = Literal["raw", "epochs"]


@dataclass(frozen=True)
class ManualReviewResult:
    mode: ResolvedReviewMode
    input_path: Path
    sidecar_path: Path
    bad_channels: list[str]
    n_annotations: int
    n_dropped_epochs: int
    cleaned_output_path: Path | None


def infer_review_mode(input_path: str | Path, mode: ReviewMode = "auto") -> ResolvedReviewMode:
    path = Path(input_path)
    if mode in {"raw", "epochs"}:
        return mode

    name = path.name.lower()
    if name.endswith("-epo.fif") or name.endswith("_epo.fif"):
        return "epochs"
    if path.suffix.lower() in {".vhdr"}:
        return "raw"
    if path.suffix.lower() == ".set":
        # .set can represent either raw or epoched data; default to raw for safety.
        return "raw"
    return "raw"


def default_sidecar_path(input_path: str | Path) -> Path:
    path = Path(input_path)
    return path.with_suffix(path.suffix + ".manual_reject.json")


def load_sidecar(sidecar_path: str | Path) -> dict[str, Any]:
    path = Path(sidecar_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid sidecar payload in {path}: expected a JSON object.")
    return data


def save_sidecar(payload: dict[str, Any], sidecar_path: str | Path) -> Path:
    path = Path(sidecar_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def apply_sidecar_to_raw(raw: mne.io.BaseRaw, payload: dict[str, Any]) -> dict[str, int]:
    bads = [str(ch) for ch in payload.get("bad_channels", [])]
    valid_bads = [ch for ch in bads if ch in raw.ch_names]

    before_bads = set(raw.info.get("bads", []))
    raw.info["bads"] = sorted(before_bads.union(valid_bads))
    n_new_bads = len(set(raw.info["bads"]) - before_bads)

    ann = payload.get("annotations", [])
    n_added_ann = 0
    if isinstance(ann, list) and ann:
        existing = {
            (round(float(on), 6), round(float(dur), 6), str(desc))
            for on, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)
        }

        add_onset: list[float] = []
        add_duration: list[float] = []
        add_description: list[str] = []
        for item in ann:
            if not isinstance(item, dict):
                continue
            onset = float(item.get("onset_s", 0.0))
            duration = float(item.get("duration_s", 0.0))
            description = str(item.get("description", "BAD_manual"))
            key = (round(onset, 6), round(duration, 6), description)
            if key in existing:
                continue
            add_onset.append(onset)
            add_duration.append(duration)
            add_description.append(description)
            existing.add(key)

        if add_onset:
            add = mne.Annotations(onset=add_onset, duration=add_duration, description=add_description)
            raw.set_annotations(raw.annotations + add)
            n_added_ann = len(add_onset)

    return {
        "bad_channels_added": int(n_new_bads),
        "annotations_added": int(n_added_ann),
    }


def apply_sidecar_to_epochs(epochs: mne.Epochs, payload: dict[str, Any]) -> dict[str, int]:
    bads = [str(ch) for ch in payload.get("bad_channels", [])]
    valid_bads = [ch for ch in bads if ch in epochs.ch_names]

    before_bads = set(epochs.info.get("bads", []))
    epochs.info["bads"] = sorted(before_bads.union(valid_bads))
    n_new_bads = len(set(epochs.info["bads"]) - before_bads)

    drop_sel = payload.get("dropped_epoch_indices", [])
    requested = sorted({int(i) for i in drop_sel if isinstance(i, int) or str(i).isdigit()})
    sel_to_pos = {int(sel): int(pos) for pos, sel in enumerate(epochs.selection.tolist())}
    drop_positions = sorted({sel_to_pos[i] for i in requested if i in sel_to_pos})
    if drop_positions:
        epochs.drop(drop_positions, reason="MANUAL_REVIEW")

    return {
        "bad_channels_added": int(n_new_bads),
        "epochs_dropped": int(len(drop_positions)),
        "epochs_requested": int(len(requested)),
    }


def _load_raw(path: Path) -> mne.io.BaseRaw:
    suf = path.suffix.lower()
    if suf == ".vhdr":
        return mne.io.read_raw_brainvision(path, preload=True)
    if suf == ".set":
        return mne.io.read_raw_eeglab(path, preload=True)
    if suf == ".fif":
        return mne.io.read_raw_fif(path, preload=True)
    raise ValueError(f"Unsupported raw format: {path.suffix}")


def _load_epochs(path: Path) -> mne.Epochs:
    suf = path.suffix.lower()
    if suf == ".fif":
        return mne.read_epochs(path, preload=True)
    if suf == ".set":
        ep = mne.read_epochs_eeglab(path, verbose="error")
        ep.load_data()
        return ep
    raise ValueError(f"Unsupported epochs format: {path.suffix}")


def _raw_payload(raw: mne.io.BaseRaw, input_path: Path) -> dict[str, Any]:
    annotations = [
        {
            "onset_s": float(on),
            "duration_s": float(dur),
            "description": str(desc),
        }
        for on, dur, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description)
    ]
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "raw",
        "input_file": str(input_path),
        "bad_channels": sorted({str(ch) for ch in raw.info.get("bads", [])}),
        "annotations": annotations,
    }


def _epochs_payload(
    epochs: mne.Epochs,
    input_path: Path,
    selection_reference: list[int],
) -> dict[str, Any]:
    selection_after = {int(i) for i in epochs.selection.tolist()}
    dropped = sorted(i for i in selection_reference if i not in selection_after)
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "epochs",
        "input_file": str(input_path),
        "bad_channels": sorted({str(ch) for ch in epochs.info.get("bads", [])}),
        "dropped_epoch_indices": dropped,
    }


def _save_cleaned(raw_or_epochs: mne.io.BaseRaw | mne.Epochs, save_path: Path) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(raw_or_epochs, mne.Epochs):
        raw_or_epochs.save(save_path, overwrite=True)
    else:
        raw_or_epochs.save(save_path, overwrite=True)
    return save_path


def review_file(
    *,
    input_path: str | Path,
    mode: ReviewMode = "auto",
    sidecar_path: str | Path | None = None,
    save_cleaned_path: str | Path | None = None,
    block: bool = True,
) -> ManualReviewResult:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    resolved_mode = infer_review_mode(path, mode=mode)
    sidecar = Path(sidecar_path) if sidecar_path is not None else default_sidecar_path(path)
    existing_payload = load_sidecar(sidecar) if sidecar.exists() else None

    if resolved_mode == "raw":
        raw = _load_raw(path)
        if existing_payload is not None:
            apply_sidecar_to_raw(raw, existing_payload)

        raw.plot(block=block)
        payload = _raw_payload(raw, path)
        save_sidecar(payload, sidecar)

        cleaned_output = Path(save_cleaned_path) if save_cleaned_path is not None else None
        if cleaned_output is not None:
            _save_cleaned(raw, cleaned_output)

        return ManualReviewResult(
            mode="raw",
            input_path=path,
            sidecar_path=sidecar,
            bad_channels=list(payload.get("bad_channels", [])),
            n_annotations=len(payload.get("annotations", [])),
            n_dropped_epochs=0,
            cleaned_output_path=cleaned_output,
        )

    epochs = _load_epochs(path)
    selection_reference = [int(i) for i in epochs.selection.tolist()]
    if existing_payload is not None:
        apply_sidecar_to_epochs(epochs, existing_payload)

    epochs.plot(block=block)
    payload = _epochs_payload(epochs, path, selection_reference=selection_reference)
    save_sidecar(payload, sidecar)

    cleaned_output = Path(save_cleaned_path) if save_cleaned_path is not None else None
    if cleaned_output is not None:
        _save_cleaned(epochs, cleaned_output)

    return ManualReviewResult(
        mode="epochs",
        input_path=path,
        sidecar_path=sidecar,
        bad_channels=list(payload.get("bad_channels", [])),
        n_annotations=0,
        n_dropped_epochs=len(payload.get("dropped_epoch_indices", [])),
        cleaned_output_path=cleaned_output,
    )
