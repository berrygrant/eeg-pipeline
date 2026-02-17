# mmn_pipeline/io_brainvision.py
from __future__ import annotations

from pathlib import Path
import mne
import pandas as pd


def read_raw_preprocess(
    raw_path: Path,
    montage: str,
    eog_chs: list[str],
    aux_chs: list[str],
    reref: str,
    l_freq: float,
    h_freq: float,
    notch: list[float] | None,
):
    suffix = raw_path.suffix.lower()
    if suffix == ".vhdr":
        raw = mne.io.read_raw_brainvision(raw_path, preload=True)
    elif suffix == ".set":
        raw = mne.io.read_raw_eeglab(raw_path, preload=True)
    else:
        raise ValueError(f"Unsupported raw file extension: {raw_path.suffix}")

    # Normalize common channel name mismatches (case and aliases)
    rename_map = {}

    def _maybe_rename(src: str, dst: str) -> None:
        if src in raw.ch_names and dst not in raw.ch_names:
            rename_map[src] = dst

    _maybe_rename("FP1", "Fp1")
    _maybe_rename("FP2", "Fp2")
    _maybe_rename("PZ", "Pz")
    _maybe_rename("CZ", "Cz")
    _maybe_rename("Mastoid L", "TP9")
    _maybe_rename("Mastoid R", "TP10")

    if rename_map:
        raw.rename_channels(rename_map)

    ch_types = {ch: "eog" for ch in eog_chs if ch in raw.ch_names}
    if ch_types:
        raw.set_channel_types(ch_types)

    drop_aux = [ch for ch in aux_chs if ch in raw.ch_names]
    if drop_aux:
        raw.drop_channels(drop_aux)

    raw.set_montage(montage, on_missing="warn")
    if eog_chs:
        eog_map = {ch: "eog" for ch in eog_chs if ch in raw.ch_names}
        if eog_map:
            raw.set_channel_types(eog_map)
    reref_mode = str(reref).strip().lower()
    if reref_mode in {"average", "avg"}:
        raw.set_eeg_reference("average", projection=False)
    elif reref_mode in {"none", "no"}:
        pass
    elif reref_mode in {"p9_p10", "tp9_tp10", "mastoids", "mastoid", "linked_mastoids", "linked"}:
        ref_pairs = [("P9", "P10"), ("TP9", "TP10")]
        ref_chs = None
        for a, b in ref_pairs:
            if a in raw.ch_names and b in raw.ch_names:
                ref_chs = [a, b]
                break
        if ref_chs is None:
            raise ValueError(
                "Requested mastoid reference but neither P9/P10 nor TP9/TP10 were found. "
                f"Available channels: {raw.ch_names}"
            )
        raw.set_eeg_reference(ref_channels=ref_chs, projection=False)
    else:
        raise ValueError(
            f"Unsupported reref mode: {reref!r} (use 'average', 'none', or 'p9_p10'/'tp9_tp10')"
        )

    if notch:
        raw.notch_filter(list(notch))
    raw.filter(l_freq=l_freq, h_freq=h_freq)
    return raw


def events_from_annotations_positions(raw):
    events, _ = mne.events_from_annotations(raw)
    return events


def parse_vmrk_markers(vmrk_path: Path) -> pd.DataFrame:
    rows = []
    with vmrk_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("Mk"):
                continue
            left, right = line.split("=", 1)
            mk_num = int(left.replace("Mk", ""))
            parts = right.split(",")
            if len(parts) < 5:
                continue
            mtype = parts[0].strip()
            desc = parts[1].strip()
            pos = int(float(parts[2]))
            size = int(float(parts[3]))
            chan = int(float(parts[4]))
            rows.append((mk_num, mtype, desc, pos, size, chan))
    return pd.DataFrame(rows, columns=["mk", "mtype", "desc", "pos", "size", "chan"])
