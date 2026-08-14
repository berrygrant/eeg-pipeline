from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PIPELINE_NAME = "eeg-pipeline"
BIDS_VERSION = "1.11.1"
RAW_EEG_EXTENSIONS = (".vhdr", ".set")

#: Directories BIDS reserves for content that is deliberately NOT BIDS-formatted,
#: plus version-control internals. Scanning them picks up files with non-BIDS
#: names -- ds003620 keeps its originals in sourcedata/sub-1/eeg/runabout1.vhdr --
#: and, in the case of derivatives/, would feed the pipeline's own outputs back in
#: as though they were raw input.
NON_BIDS_DIRS = frozenset({
    "sourcedata", "derivatives", "code", "stimuli",
    ".git", ".datalad", ".annex", ".github",
})
ENTITY_ORDER = ("sub", "ses", "task", "acq", "run", "recording")
DERIVATIVE_EXTENSIONS = (".fif", ".tsv", ".csv", ".parquet")


@dataclass(frozen=True)
class BIDSRecording:
    bids_root: Path
    raw_path: Path
    entities: dict[str, str]
    datatype: str = "eeg"

    @property
    def basename(self) -> str:
        return build_bids_basename(self.entities, suffix="eeg")

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
        ses = self.session_id
        return None if ses is None else f"ses-{ses}"

    @property
    def task_id(self) -> str | None:
        return self.entities.get("task")

    @property
    def run_id(self) -> str | None:
        return self.entities.get("run")

    @property
    def events_path(self) -> Path:
        return self.raw_path.with_name(f"{self.basename.replace('_eeg', '')}_events.tsv")

    @property
    def events_json_path(self) -> Path:
        return self.events_path.with_suffix(".json")

    @property
    def raw_json_path(self) -> Path:
        return self.raw_path.with_suffix(".json")

    @property
    def relative_raw_path(self) -> str:
        return self.raw_path.relative_to(self.bids_root).as_posix()

    @property
    def relative_events_path(self) -> str:
        return self.events_path.relative_to(self.bids_root).as_posix()


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


def parse_bids_entities(path: Path) -> dict[str, str]:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Path is not a BIDS EEG file: {path}")

    entities: dict[str, str] = {}
    for token in parts[:-1]:
        if "-" not in token:
            continue
        key, value = token.split("-", 1)
        if key in ENTITY_ORDER and value:
            entities[key] = value

    if parts[-1] != "eeg":
        raise ValueError(f"Expected BIDS EEG suffix in filename: {path.name}")
    if "sub" not in entities:
        raise ValueError(f"Missing sub entity in filename: {path.name}")

    sub_dir = next((part for part in path.parts if part.startswith("sub-")), None)
    ses_dir = next((part for part in path.parts if part.startswith("ses-")), None)
    if sub_dir and sub_dir != f"sub-{entities['sub']}":
        raise ValueError(f"Subject directory does not match filename: {path}")
    if ses_dir:
        ses_value = ses_dir[4:]
        if entities.get("ses", ses_value) != ses_value:
            raise ValueError(f"Session directory does not match filename: {path}")
        entities.setdefault("ses", ses_value)

    return entities


def discover_bids_eeg_recordings(
    bids_root: Path,
    *,
    subjects: Iterable[str] | None = None,
    sessions: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    runs: Iterable[str] | None = None,
) -> list[BIDSRecording]:
    bids_root = Path(bids_root)
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root not found: {bids_root}")

    subject_filter = _normalize_filter_values(subjects, "sub")
    session_filter = _normalize_filter_values(sessions, "ses")
    task_filter = _normalize_filter_values(tasks, "task")
    run_filter = _normalize_filter_values(runs, "run")

    files: list[Path] = []
    for ext in RAW_EEG_EXTENSIONS:
        files.extend(
            p
            for p in bids_root.rglob(f"*{ext}")
            if p.is_file() and not (set(p.parts) & NON_BIDS_DIRS)
        )

    recordings: list[BIDSRecording] = []
    unparseable: list[Path] = []
    for raw_path in sorted(files):
        try:
            entities = parse_bids_entities(raw_path)
        except ValueError:
            # Inside the BIDS tree proper (the reserved directories are already
            # excluded above), an unparseable name is worth reporting rather than
            # dropping quietly -- but it must not abort discovery for the whole
            # dataset, which is what raising here used to do.
            unparseable.append(raw_path)
            continue
        if subject_filter and entities["sub"] not in subject_filter:
            continue
        if session_filter and entities.get("ses") not in session_filter:
            continue
        if task_filter and entities.get("task") not in task_filter:
            continue
        if run_filter and entities.get("run") not in run_filter:
            continue
        recordings.append(BIDSRecording(bids_root=bids_root, raw_path=raw_path, entities=entities))

    if unparseable:
        print(
            f"[WARN] {len(unparseable)} file(s) under {bids_root} have non-BIDS names "
            "and were skipped; they are not in a reserved directory, so check whether "
            "they should have been processed:"
        )
        for path in unparseable[:10]:
            print(f"          {path.relative_to(bids_root)}")
        if len(unparseable) > 10:
            print(f"          ... and {len(unparseable) - 10} more")

    return recordings


def filter_derivative_paths(
    paths: Iterable[Path],
    *,
    subjects: Iterable[str] | None = None,
    sessions: Iterable[str] | None = None,
    tasks: Iterable[str] | None = None,
    runs: Iterable[str] | None = None,
) -> list[Path]:
    """Filter derivative files by BIDS entities.

    Applies the same entity-filter semantics as :func:`discover_bids_eeg_recordings`
    (bare or prefixed values, e.g. ``01`` or ``sub-01``) to files already inside a
    derivatives tree, whose names carry ``desc-``/``suffix`` tokens rather than the
    raw ``_eeg`` suffix. With no filters supplied the input order is preserved
    unchanged.

    This is what lets a single-subject invocation (``--get_metrics --subjects
    sub-01``) touch only that subject's derivatives, which in turn makes it safe to
    fan the metrics stage out across concurrent jobs.
    """
    subject_filter = _normalize_filter_values(subjects, "sub")
    session_filter = _normalize_filter_values(sessions, "ses")
    task_filter = _normalize_filter_values(tasks, "task")
    run_filter = _normalize_filter_values(runs, "run")

    if not any((subject_filter, session_filter, task_filter, run_filter)):
        return [Path(p) for p in paths]

    kept: list[Path] = []
    for path in paths:
        path = Path(path)
        try:
            entities = parse_bids_entities_like_name(path.stem)
        except ValueError:
            # Not a BIDS-like derivative name; a filter cannot confirm a match.
            continue
        if subject_filter and entities.get("sub") not in subject_filter:
            continue
        if session_filter and entities.get("ses") not in session_filter:
            continue
        if task_filter and entities.get("task") not in task_filter:
            continue
        if run_filter and entities.get("run") not in run_filter:
            continue
        kept.append(path)
    return kept


def build_bids_basename(
    entities: Mapping[str, str],
    *,
    suffix: str,
    desc: str | None = None,
) -> str:
    tokens: list[str] = []
    for key in ENTITY_ORDER:
        value = entities.get(key)
        if value:
            tokens.append(f"{key}-{value}")
    if desc:
        tokens.append(f"desc-{desc}")
    tokens.append(suffix)
    return "_".join(tokens)


def ensure_derivatives_dataset(
    derivatives_root: Path,
    *,
    source_dataset: Path | None = None,
    name: str = PIPELINE_NAME,
    pipeline_version: str | None = None,
) -> Path:
    derivatives_root = Path(derivatives_root)
    dataset_root = derivatives_root / name
    dataset_root.mkdir(parents=True, exist_ok=True)

    generated_by: list[dict[str, Any]] = [{"Name": name}]
    if pipeline_version:
        generated_by[0]["Version"] = pipeline_version

    dataset_description: dict[str, Any] = {
        "Name": f"{name} derivatives",
        "BIDSVersion": BIDS_VERSION,
        "DatasetType": "derivative",
        "GeneratedBy": generated_by,
    }
    if source_dataset is not None:
        dataset_description["SourceDatasets"] = [{"URL": Path(source_dataset).resolve().as_uri()}]

    write_json(dataset_root / "dataset_description.json", dataset_description)
    return dataset_root


def subject_derivatives_dir(dataset_root: Path, entities: Mapping[str, str], datatype: str = "eeg") -> Path:
    parts = [Path(dataset_root), f"sub-{entities['sub']}"]
    if entities.get("ses"):
        parts.append(f"ses-{entities['ses']}")
    parts.append(datatype)
    out = Path(parts[0])
    for part in parts[1:]:
        out = out / part
    out.mkdir(parents=True, exist_ok=True)
    return out


def dataset_derivatives_dir(dataset_root: Path, datatype: str = "eeg") -> Path:
    out = Path(dataset_root) / datatype
    out.mkdir(parents=True, exist_ok=True)
    return out


def subject_derivative_path(
    dataset_root: Path,
    entities: Mapping[str, str],
    *,
    suffix: str,
    extension: str,
    desc: str | None = None,
    datatype: str = "eeg",
) -> Path:
    if not extension.startswith("."):
        extension = f".{extension}"
    out_dir = subject_derivatives_dir(dataset_root, entities, datatype=datatype)
    return out_dir / f"{build_bids_basename(entities, suffix=suffix, desc=desc)}{extension}"


def dataset_derivative_path(
    dataset_root: Path,
    *,
    suffix: str,
    extension: str,
    desc: str | None = None,
    entities: Mapping[str, str] | None = None,
    datatype: str = "eeg",
) -> Path:
    if not extension.startswith("."):
        extension = f".{extension}"
    out_dir = dataset_derivatives_dir(dataset_root, datatype=datatype)
    basename = build_bids_basename(entities or {}, suffix=suffix, desc=desc)
    return out_dir / f"{basename}{extension}"


def sidecar_json_path(data_path: Path) -> Path:
    if data_path.suffix == ".json":
        return data_path
    return data_path.with_suffix(".json")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def derivative_sidecar_path(data_path: Path) -> Path:
    return sidecar_json_path(data_path)


def source_basename_from_derivative_path(path: Path) -> str:
    entities = parse_bids_entities_like_name(path.stem)
    return build_bids_basename(entities, suffix="eeg")


def parse_bids_entities_like_name(stem: str) -> dict[str, str]:
    parts = stem.split("_")
    entities: dict[str, str] = {}
    for token in parts:
        if "-" not in token:
            continue
        key, value = token.split("-", 1)
        if key in ENTITY_ORDER and value:
            entities[key] = value
    if "sub" not in entities:
        raise ValueError(f"Missing sub entity in derivative stem: {stem}")
    return entities


def validate_bids_dataset(bids_root: Path) -> list[str]:
    errors: list[str] = []
    bids_root = Path(bids_root)
    dataset_description = bids_root / "dataset_description.json"
    if not dataset_description.exists():
        errors.append(f"Missing dataset_description.json in {bids_root}")
    else:
        try:
            data = json.loads(dataset_description.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"Invalid JSON in {dataset_description}: {exc}")
            data = {}
        if "Name" not in data:
            errors.append(f"Missing Name in {dataset_description}")
        if "BIDSVersion" not in data:
            errors.append(f"Missing BIDSVersion in {dataset_description}")

    recordings = discover_bids_eeg_recordings(bids_root)
    if not recordings:
        errors.append(f"No BIDS EEG recordings found in {bids_root}")
        return errors

    for recording in recordings:
        if not recording.events_path.exists():
            errors.append(f"Missing events.tsv for {recording.raw_path.relative_to(bids_root)}")

    return errors


def validate_derivatives_dataset(dataset_root: Path) -> list[str]:
    errors: list[str] = []
    dataset_root = Path(dataset_root)
    dataset_description = dataset_root / "dataset_description.json"
    if not dataset_description.exists():
        errors.append(f"Missing dataset_description.json in {dataset_root}")
        return errors

    try:
        data = json.loads(dataset_description.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in {dataset_description}: {exc}")
        return errors

    for key in ("Name", "BIDSVersion", "DatasetType", "GeneratedBy"):
        if key not in data:
            errors.append(f"Missing {key} in {dataset_description}")
    if data.get("DatasetType") != "derivative":
        errors.append(f"DatasetType must be 'derivative' in {dataset_description}")

    derivative_files = [
        p
        for p in dataset_root.rglob("*")
        if p.is_file()
        and p.name != "dataset_description.json"
        and p.suffix in DERIVATIVE_EXTENSIONS
    ]
    if not derivative_files:
        errors.append(f"No derivative files found in {dataset_root}")
        return errors

    for path in derivative_files:
        sidecar = derivative_sidecar_path(path)
        if not sidecar.exists():
            errors.append(f"Missing JSON sidecar for {path.relative_to(dataset_root)}")

    return errors
