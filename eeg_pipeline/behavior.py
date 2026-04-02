from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import derive_metadata_from_condition_map, derive_metadata_v1


BIDS_CODE_COLUMNS = ("value", "trial_code", "event_code", "trigger", "stim_code", "code")


@dataclass
class BehavioralEvents:
    source: str
    source_path: Path
    sidecar: dict[str, Any]
    codes_all: np.ndarray
    codes: np.ndarray
    metadata_all: pd.DataFrame
    metadata: pd.DataFrame
    samples: np.ndarray | None


def read_eventcodes_from_subject_csv(subject_csv: Path) -> np.ndarray:
    df = pd.read_csv(subject_csv)
    if "EventCode" not in df.columns:
        raise ValueError(
            f"'EventCode' column not found in {subject_csv}. "
            f"Found columns: {list(df.columns)}"
        )
    codes = df["EventCode"].to_numpy()
    if not np.issubdtype(codes.dtype, np.integer):
        codes = codes.astype(int)
    return codes


def filter_codes(codes: np.ndarray, keep_codes: list[int] | None) -> np.ndarray:
    if keep_codes is None or len(keep_codes) == 0:
        return codes
    keep = np.isin(codes, np.asarray(keep_codes, dtype=int))
    return codes[keep]


def behavior_keep_mask(codes: np.ndarray, keep_codes: list[int] | None) -> np.ndarray:
    if keep_codes is None or len(keep_codes) == 0:
        return np.ones(len(codes), dtype=bool)
    return np.isin(codes, np.asarray(keep_codes, dtype=int))


def read_bids_events_table(events_tsv: Path) -> pd.DataFrame:
    if not events_tsv.exists():
        raise FileNotFoundError(f"BIDS events file not found: {events_tsv}")
    return pd.read_csv(events_tsv, sep="\t")


def read_bids_events_sidecar(events_json: Path) -> dict[str, Any]:
    if not events_json.exists():
        return {}
    return json.loads(events_json.read_text(encoding="utf-8"))


def extract_codes_from_bids_events(
    events_df: pd.DataFrame,
    *,
    condition_map: dict[str, list[int]] | None = None,
) -> np.ndarray:
    for column in BIDS_CODE_COLUMNS:
        if column not in events_df.columns:
            continue
        codes = pd.to_numeric(events_df[column], errors="raise").astype(int)
        return codes.to_numpy(dtype=int)

    if "trial_type" in events_df.columns and condition_map:
        reverse_map: dict[str, int] = {}
        for name, values in condition_map.items():
            if len(values) != 1:
                raise ValueError(
                    "events.condition_map must map each label to a single numeric code "
                    "when BIDS events rely on trial_type labels."
                )
            reverse_map[str(name)] = int(values[0])
        try:
            return events_df["trial_type"].map(reverse_map).astype(int).to_numpy(dtype=int)
        except Exception as exc:
            raise ValueError(
                "Could not derive numeric event codes from trial_type. "
                "Provide a numeric events column such as 'value' or a matching events.condition_map."
            ) from exc

    raise ValueError(
        "BIDS events must include a numeric code column "
        f"({', '.join(BIDS_CODE_COLUMNS)}) or trial_type labels plus events.condition_map."
    )


def _normalize_bids_events_metadata(
    events_df: pd.DataFrame,
    *,
    codes: np.ndarray,
    token_map: dict[str, str] | None = None,
    condition_map: dict[str, list[int]] | None = None,
) -> pd.DataFrame:
    metadata = events_df.copy().reset_index(drop=True)
    metadata["code"] = np.asarray(codes, dtype=int)

    if "trial_type" not in metadata.columns and condition_map:
        reverse_map = {int(values[0]): str(name) for name, values in condition_map.items() if len(values) == 1}
        metadata["trial_type"] = metadata["code"].map(reverse_map).fillna("UNKNOWN")

    if condition_map:
        cond_meta = derive_metadata_from_condition_map(metadata["code"].tolist(), condition_map)
        for column in cond_meta.columns:
            if column not in metadata.columns:
                metadata[column] = cond_meta[column]
    else:
        try:
            schema_meta = derive_metadata_v1(metadata["code"].tolist(), token_map=token_map)
        except Exception:
            schema_meta = None
        if schema_meta is not None:
            for column in schema_meta.columns:
                if column not in metadata.columns:
                    metadata[column] = schema_meta[column]

    return metadata


def _extract_bids_samples(events_df: pd.DataFrame) -> np.ndarray | None:
    if "sample" not in events_df.columns:
        return None
    samples = pd.to_numeric(events_df["sample"], errors="coerce")
    if samples.isna().any():
        return None
    return samples.astype(int).to_numpy(dtype=int)


def _resolve_subject_csv_path(subject_id: str, fallback_dir: Path) -> Path:
    candidates = [
        fallback_dir / f"sub-{subject_id}.csv",
        fallback_dir / f"subject-{subject_id}.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_behavioral_events(
    *,
    events_tsv: Path,
    events_json: Path,
    subject_id: str,
    keep_codes: list[int] | None,
    token_map: dict[str, str] | None = None,
    condition_map: dict[str, list[int]] | None = None,
    csv_path: Path | None = None,
    csv_fallback_dir: Path | None = None,
) -> BehavioralEvents:
    if events_tsv.exists():
        events_df = read_bids_events_table(events_tsv)
        sidecar = read_bids_events_sidecar(events_json)
        codes_all = extract_codes_from_bids_events(events_df, condition_map=condition_map)
        keep_mask = behavior_keep_mask(codes_all, keep_codes)
        metadata_all = _normalize_bids_events_metadata(
            events_df,
            codes=codes_all,
            token_map=token_map,
            condition_map=condition_map,
        )
        samples = _extract_bids_samples(metadata_all)
        if samples is not None:
            samples = samples[keep_mask]
        return BehavioralEvents(
            source="bids_events",
            source_path=events_tsv,
            sidecar=sidecar,
            codes_all=codes_all,
            codes=np.asarray(codes_all[keep_mask], dtype=int),
            metadata_all=metadata_all,
            metadata=metadata_all.loc[keep_mask].reset_index(drop=True),
            samples=samples,
        )

    if csv_path is None and csv_fallback_dir is None:
        raise FileNotFoundError(f"Missing BIDS events file: {events_tsv}")

    subject_csv = None
    if csv_path is not None and csv_path.exists():
        subject_csv = csv_path
    elif csv_fallback_dir is not None:
        subject_csv = _resolve_subject_csv_path(subject_id, csv_fallback_dir)
    elif csv_path is not None:
        subject_csv = csv_path
    if not subject_csv.exists():
        raise FileNotFoundError(
            f"Missing BIDS events file {events_tsv} and CSV fallback file {subject_csv}"
        )

    codes_all = read_eventcodes_from_subject_csv(subject_csv)
    keep_mask = behavior_keep_mask(codes_all, keep_codes)
    if condition_map:
        metadata_all = derive_metadata_from_condition_map(codes_all.tolist(), condition_map)
    else:
        metadata_all = derive_metadata_v1(codes_all.tolist(), token_map=token_map)
    return BehavioralEvents(
        source="csv_fallback",
        source_path=subject_csv,
        sidecar={},
        codes_all=codes_all,
        codes=np.asarray(codes_all[keep_mask], dtype=int),
        metadata_all=metadata_all,
        metadata=metadata_all.loc[keep_mask].reset_index(drop=True),
        samples=None,
    )
