# mmn_pipeline/behavior.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from .schema import SchemaV1Config


def resolve_subject_csv_path(subject_csv_dir: Path, subj_num: str, subj_stem: str | None = None) -> Path:
    num = str(subj_num).strip()
    stem = str(subj_stem).strip() if subj_stem else ""

    candidates = [
        f"subject-{num}.csv",
        f"S{num}-eventcodes.csv",
        f"s{num}-eventcodes.csv",
        f"S{num}.csv",
        f"s{num}.csv",
    ]
    if stem:
        candidates.extend(
            [
                f"{stem}-eventcodes.csv",
                f"{stem}.csv",
                f"{stem.lower()}-eventcodes.csv",
                f"{stem.lower()}.csv",
            ]
        )

    seen: set[str] = set()
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        candidate = Path(subject_csv_dir) / name
        if candidate.exists():
            return candidate

    return Path(subject_csv_dir) / f"subject-{num}.csv"


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


def write_eventcodes_csv(
    subject_csv: Path,
    out_csv: Path,
) -> int:
    codes = read_eventcodes_from_subject_csv(subject_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"EventCode": codes.astype(int)}).to_csv(out_csv, index=False)
    return int(len(codes))


def clean_eventcodes(
    codes: np.ndarray,
    mode: str | None = None,
    cfg: SchemaV1Config | None = None,
) -> tuple[np.ndarray, dict]:
    mode_norm = "none" if mode in (None, "") else str(mode).strip().lower()
    codes_arr = np.asarray(codes, dtype=int)
    diag = {
        "eventcode_cleanup_mode": mode_norm,
        "eventcode_cleanup_removed": 0,
        "eventcode_cleanup_runs": 0,
    }

    if mode_norm == "none":
        return codes_arr, diag

    if mode_norm != "mprocacc_thesis":
        raise ValueError(
            f"Unsupported eventcode cleanup mode: {mode!r} "
            "(use 'none' or 'mprocacc_thesis')."
        )

    if codes_arr.size == 0:
        return codes_arr, diag

    if cfg is None:
        cfg = SchemaV1Config()
    main_a = (set(cfg.full_A) | set(cfg.reduced_A)) - set(cfg.practice_A)

    A = codes_arr // 100
    B = (codes_arr // 10) % 10

    keep = np.ones(len(codes_arr), dtype=bool)
    removed_runs = 0
    start = 0
    cur_a = int(A[0])
    cur_b = int(B[0])

    for i in range(1, len(codes_arr) + 1):
        boundary = i == len(codes_arr)
        if not boundary:
            boundary = int(A[i]) != cur_a or int(B[i]) != cur_b
        if not boundary:
            continue

        run_len = i - start
        if cur_b == 0 and cur_a in main_a:
            if run_len != 3:
                raise ValueError(
                    "mprocacc_thesis cleanup expected main-block buffer runs of length 3, "
                    f"but found A={cur_a}, B={cur_b}, run_len={run_len} at event index {start + 1}."
                )
            keep[start] = False
            removed_runs += 1

        if i < len(codes_arr):
            start = i
            cur_a = int(A[i])
            cur_b = int(B[i])

    cleaned = codes_arr[keep]
    diag["eventcode_cleanup_removed"] = int(len(codes_arr) - len(cleaned))
    diag["eventcode_cleanup_runs"] = int(removed_runs)
    return cleaned, diag


def filter_codes(codes: np.ndarray, keep_codes: list[int] | None) -> np.ndarray:
    if keep_codes is None or len(keep_codes) == 0:
        return codes
    keep = np.isin(codes, np.asarray(keep_codes, dtype=int))
    return codes[keep]
