# mmn_pipeline/io_brainvision.py
from __future__ import annotations

from pathlib import Path
import mne
import pandas as pd


def read_raw_preprocess(
    vhdr_path: Path,
    montage: str,
    eog_chs: list[str],
    aux_chs: list[str],
    l_freq: float,
    h_freq: float,
    notch: list[float] | None,
):
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True)

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
    raw.set_eeg_reference("average", projection=False)

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