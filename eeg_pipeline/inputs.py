from __future__ import annotations

import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import mne
import pandas as pd

from .align import align_marker_positions_to_codes
from .behavior import filter_codes, read_eventcodes_from_subject_csv
from .bids import (
    BIDS_VERSION,
    PIPELINE_NAME,
    build_bids_basename,
    discover_bids_eeg_recordings,
    parse_bids_entities_like_name,
    write_json,
)
from .io_brainvision import events_from_annotations_positions


@dataclass(frozen=True)
class PipelineRecording:
    source_type: str
    source_root: Path
    raw_path: Path
    entities: dict[str, str]
    behavior_path: Path | None = None
    behavior_json_path: Path | None = None
    behavior_kind: str = "none"

    @property
    def subject_id(self) -> str:
        return self.entities["sub"]

    @property
    def subject_label(self) -> str:
        return f"sub-{self.subject_id}"

    @property
    def session_id(self) -> str | None:
        return self.entities.get("ses")

    @property
    def session_label(self) -> str | None:
        if self.session_id is None:
            return None
        return f"ses-{self.session_id}"

    @property
    def task_id(self) -> str | None:
        return self.entities.get("task")

    @property
    def run_id(self) -> str | None:
        return self.entities.get("run")

    @property
    def relative_raw_path(self) -> str:
        try:
            return self.raw_path.relative_to(self.source_root).as_posix()
        except ValueError:
            return str(self.raw_path)

    @property
    def relative_behavior_path(self) -> str | None:
        if self.behavior_path is None:
            return None
        try:
            return self.behavior_path.relative_to(self.source_root).as_posix()
        except ValueError:
            return str(self.behavior_path)


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


def _normalize_filter_values(values: Iterable[str] | None, prefix: str) -> set[str] | None:
    if values is None:
        return None
    normalized: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if text.startswith(f"{prefix}-"):
            text = text[len(prefix) + 1 :]
        normalized.add(text)
    return normalized or None


def _resolve_subject_csv_path(subject_id: str, subject_csv_dir: Path | None) -> Path | None:
    if subject_csv_dir is None:
        return None
    candidates = [
        subject_csv_dir / f"subject-{subject_id}.csv",
        subject_csv_dir / f"sub-{subject_id}.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _legacy_entities_from_path(raw_path: Path, task_label: str | None) -> dict[str, str]:
    entities: dict[str, str] = {}
    try:
        parsed = parse_bids_entities_like_name(raw_path.stem)
    except Exception:
        parsed = {}
    entities.update(parsed)
    entities.setdefault("sub", subject_number_from_stem(raw_path.stem))
    if task_label and "task" not in entities:
        entities["task"] = str(task_label)

    ses_match = re.search(r"(?:^|[_-])ses[-_]?([A-Za-z0-9]+)", raw_path.stem, flags=re.IGNORECASE)
    run_match = re.search(r"(?:^|[_-])run[-_]?([A-Za-z0-9]+)", raw_path.stem, flags=re.IGNORECASE)
    if ses_match and "ses" not in entities:
        entities["ses"] = ses_match.group(1)
    if run_match and "run" not in entities:
        entities["run"] = run_match.group(1)
    return entities


def discover_legacy_recordings(
    raw_dir: Path,
    *,
    subject_csv_dir: Path | None,
    subjects: Iterable[str] | None = None,
    sessions: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    runs: Iterable[str] | None = None,
    task_label: str | None = None,
) -> list[PipelineRecording]:
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Legacy raw_dir not found: {raw_dir}")

    subject_filter = _normalize_filter_values(subjects, "sub")
    session_filter = _normalize_filter_values(sessions, "ses")
    task_filter = _normalize_filter_values(tasks, "task")
    run_filter = _normalize_filter_values(runs, "run")

    raw_files: list[Path] = []
    for pattern in ("*.vhdr", "*.set"):
        raw_files.extend(
            p for p in raw_dir.rglob(pattern) if p.is_file() and ".git" not in p.parts
        )

    recordings: list[PipelineRecording] = []
    for raw_path in sorted(raw_files):
        entities = _legacy_entities_from_path(raw_path, task_label=task_label)
        if subject_filter and entities["sub"] not in subject_filter:
            continue
        if session_filter and entities.get("ses") not in session_filter:
            continue
        if task_filter and entities.get("task") not in task_filter:
            continue
        if run_filter and entities.get("run") not in run_filter:
            continue
        behavior_path = _resolve_subject_csv_path(entities["sub"], subject_csv_dir)
        recordings.append(
            PipelineRecording(
                source_type="legacy",
                source_root=raw_dir,
                raw_path=raw_path,
                entities=entities,
                behavior_path=behavior_path,
                behavior_json_path=None,
                behavior_kind="csv",
            )
        )
    return recordings


def discover_pipeline_recordings(
    *,
    mode: str,
    bids_root: Path | None,
    raw_dir: Path | None,
    subject_csv_dir: Path | None,
    subjects: Iterable[str] | None = None,
    sessions: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    runs: Iterable[str] | None = None,
    task_label: str | None = None,
) -> list[PipelineRecording]:
    mode = str(mode or "bids").lower()
    if mode == "legacy":
        if raw_dir is None:
            raise ValueError("Legacy mode requires paths.raw_dir or --raw_dir.")
        return discover_legacy_recordings(
            Path(raw_dir),
            subject_csv_dir=None if subject_csv_dir is None else Path(subject_csv_dir),
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            runs=runs,
            task_label=task_label,
        )

    if bids_root is None:
        raise ValueError("BIDS mode requires paths.bids_root or --bids_root.")
    recordings = discover_bids_eeg_recordings(
        Path(bids_root),
        subjects=subjects,
        sessions=sessions,
        tasks=tasks,
        runs=runs,
    )
    return [
        PipelineRecording(
            source_type="bids",
            source_root=recording.bids_root,
            raw_path=recording.raw_path,
            entities=dict(recording.entities),
            behavior_path=recording.events_path,
            behavior_json_path=recording.events_json_path,
            behavior_kind="bids_events",
        )
        for recording in recordings
    ]


def _copy_text_with_replacements(source: Path, target: Path, replacements: dict[str, str]) -> None:
    text = source.read_text(encoding="utf-8", errors="replace")
    for old, new in replacements.items():
        text = text.replace(old, new)
    target.write_text(text, encoding="utf-8")


def _copy_legacy_raw_to_bids(recording: PipelineRecording, target_raw: Path) -> None:
    target_raw.parent.mkdir(parents=True, exist_ok=True)
    suffix = recording.raw_path.suffix.lower()
    if suffix == ".vhdr":
        source_vhdr = recording.raw_path
        source_vmrk = recording.raw_path.with_suffix(".vmrk")
        source_eeg = recording.raw_path.with_suffix(".eeg")
        target_vmrk = target_raw.with_suffix(".vmrk")
        target_eeg = target_raw.with_suffix(".eeg")
        shutil.copy2(source_eeg, target_eeg)
        replacements = {
            source_vmrk.name: target_vmrk.name,
            source_eeg.name: target_eeg.name,
        }
        _copy_text_with_replacements(source_vhdr, target_raw, replacements)
        _copy_text_with_replacements(source_vmrk, target_vmrk, replacements)
        return

    shutil.copy2(recording.raw_path, target_raw)
    if suffix == ".set":
        source_fdt = recording.raw_path.with_suffix(".fdt")
        if source_fdt.exists():
            shutil.copy2(source_fdt, target_raw.with_suffix(".fdt"))
            shutil.copy2(source_fdt, target_raw.parent / source_fdt.name)


def _read_raw_minimal(raw_path: Path):
    suffix = raw_path.suffix.lower()
    if suffix == ".vhdr":
        return mne.io.read_raw_brainvision(raw_path, preload=False)
    if suffix == ".set":
        return mne.io.read_raw_eeglab(raw_path, preload=False)
    raise ValueError(f"Unsupported raw file extension: {raw_path.suffix}")


def _channels_tsv(raw) -> pd.DataFrame:
    ch_types = raw.get_channel_types()
    rows = []
    for name, ch_type in zip(raw.ch_names, ch_types, strict=True):
        rows.append(
            {
                "name": name,
                "type": str(ch_type).upper(),
                "units": "V",
                "status": "good",
                "status_description": "",
            }
        )
    return pd.DataFrame(rows)


def _events_sidecar() -> dict[str, object]:
    return {
        "onset": {"Description": "Event onset in seconds from recording start."},
        "duration": {"Description": "Event duration in seconds."},
        "sample": {"Description": "Event sample index in the EEG recording."},
        "value": {"Description": "Numeric event code imported from the legacy behavioral CSV."},
        "trial_type": {"Description": "Condition label derived during legacy-to-BIDS conversion when available."},
    }


def convert_legacy_recordings_to_bids(
    recordings: list[PipelineRecording],
    *,
    bids_root: Path,
    task_label: str | None,
    keep_codes: list[int] | None,
    standard_codes: list[int] | None,
    deviant_codes: list[int] | None,
    drop_eeg_markers_by_gap_s: float | None,
    auto_drop_to_count: bool,
    overwrite: bool = True,
) -> list[PipelineRecording]:
    bids_root = Path(bids_root)
    bids_root.mkdir(parents=True, exist_ok=True)
    write_json(
        bids_root / "dataset_description.json",
        {
            "Name": f"{PIPELINE_NAME} imported BIDS dataset",
            "BIDSVersion": BIDS_VERSION,
            "DatasetType": "raw",
            "GeneratedBy": [{"Name": PIPELINE_NAME}],
        },
    )

    participants: list[dict[str, str]] = []
    converted: list[PipelineRecording] = []
    std_set = set(int(v) for v in (standard_codes or []))
    dev_set = set(int(v) for v in (deviant_codes or []))

    for recording in recordings:
        entities = dict(recording.entities)
        if task_label and "task" not in entities:
            entities["task"] = task_label

        datatype_dir = bids_root / f"sub-{entities['sub']}"
        if entities.get("ses"):
            datatype_dir = datatype_dir / f"ses-{entities['ses']}"
        datatype_dir = datatype_dir / "eeg"
        datatype_dir.mkdir(parents=True, exist_ok=True)

        basename = build_bids_basename(entities, suffix="eeg")
        target_raw = datatype_dir / f"{basename}{recording.raw_path.suffix.lower()}"
        if target_raw.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing converted file: {target_raw}")
        _copy_legacy_raw_to_bids(recording, target_raw)

        raw = _read_raw_minimal(recording.raw_path)
        write_json(
            target_raw.with_suffix(".json"),
            {
                "TaskName": entities.get("task", "task"),
                "SamplingFrequency": float(raw.info["sfreq"]),
                "PowerLineFrequency": 60,
                "EEGReference": "n/a",
            },
        )

        channels_base = basename.replace("_eeg", "")
        _channels_tsv(raw).to_csv(datatype_dir / f"{channels_base}_channels.tsv", sep="\t", index=False)

        target_events = datatype_dir / f"{channels_base}_events.tsv"
        target_events_json = datatype_dir / f"{channels_base}_events.json"
        if recording.behavior_path and recording.behavior_path.exists():
            codes_all = read_eventcodes_from_subject_csv(recording.behavior_path)
            codes = filter_codes(codes_all, keep_codes)
            events_ann = events_from_annotations_positions(raw)
            markers_pos = events_ann[:, 0].copy()
            markers_aligned, _ = align_marker_positions_to_codes(
                markers_pos=markers_pos,
                sfreq=float(raw.info["sfreq"]),
                codes=codes,
                gap_s=drop_eeg_markers_by_gap_s,
                auto_drop_to_count=bool(auto_drop_to_count),
            )
            event_rows = []
            # strict=False: with auto_drop_to_count disabled, aligned markers
            # may exceed codes; pair each code with its marker and ignore extras.
            for sample, code in zip(markers_aligned.tolist(), codes.tolist(), strict=False):
                if code in std_set:
                    trial_type = "Standard"
                elif code in dev_set:
                    trial_type = "Deviant"
                else:
                    trial_type = f"code_{code}"
                event_rows.append(
                    {
                        "onset": float(sample) / float(raw.info["sfreq"]),
                        "duration": 0.0,
                        "sample": int(sample),
                        "trial_type": trial_type,
                        "value": int(code),
                    }
                )
            pd.DataFrame(event_rows).to_csv(target_events, sep="\t", index=False)
            write_json(target_events_json, _events_sidecar())
            behavior_path = target_events
            behavior_json_path = target_events_json
            behavior_kind = "bids_events"
        else:
            behavior_path = None
            behavior_json_path = None
            behavior_kind = "none"

        participants.append({"participant_id": f"sub-{entities['sub']}"})
        converted.append(
            PipelineRecording(
                source_type="bids",
                source_root=bids_root,
                raw_path=target_raw,
                entities=entities,
                behavior_path=behavior_path,
                behavior_json_path=behavior_json_path,
                behavior_kind=behavior_kind,
            )
        )

    if participants:
        (
            pd.DataFrame(participants)
            .drop_duplicates()
            .sort_values("participant_id")
            .to_csv(bids_root / "participants.tsv", sep="\t", index=False)
        )

    return converted
