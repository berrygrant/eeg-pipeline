from pathlib import Path

import numpy as np
import pytest

import eeg_pipeline.io_brainvision as io_mod


class DummyRaw:
    def __init__(self, ch_names):
        self.ch_names = list(ch_names)
        self.rename_calls = []
        self.channel_type_calls = []
        self.drop_calls = []
        self.montage_calls = []
        self.reference_calls = []
        self.notch_calls = []
        self.filter_calls = []

    def rename_channels(self, rename_map):
        self.rename_calls.append(dict(rename_map))
        self.ch_names = [rename_map.get(ch, ch) for ch in self.ch_names]

    def set_channel_types(self, mapping, on_unit_change=None):
        self.channel_type_calls.append((dict(mapping), on_unit_change))

    def drop_channels(self, channels):
        self.drop_calls.append(list(channels))
        self.ch_names = [ch for ch in self.ch_names if ch not in channels]

    def set_montage(self, montage, on_missing="warn"):
        self.montage_calls.append((montage, on_missing))

    def set_eeg_reference(self, ref_channels="average", projection=False):
        self.reference_calls.append((ref_channels, projection))

    def notch_filter(self, freqs, n_jobs=1):
        # Record n_jobs so tests can assert it is actually forwarded: filtering
        # is the main within-subject parallelism lever, and silently dropping
        # n_jobs here would leave many-channel runs single-threaded.
        self.notch_calls.append((list(freqs), n_jobs))

    def filter(self, l_freq, h_freq, n_jobs=1):
        self.filter_calls.append((l_freq, h_freq, n_jobs))
        return self


def test_read_raw_preprocess_rejects_unknown_file_extensions(tmp_path: Path):
    with pytest.raises(ValueError, match="Unsupported raw file extension"):
        io_mod.read_raw_preprocess(
            tmp_path / "sample.txt",
            montage="standard_1020",
            eog_chs=[],
            aux_chs=[],
            reref="average",
            l_freq=0.1,
            h_freq=30.0,
            notch=None,
        )


def test_read_raw_preprocess_brainvision_loader_renames_drops_and_average_references(monkeypatch, tmp_path: Path):
    raw = DummyRaw(["FP1", "PZ", "EOG1", "AUX", "Cz"])
    monkeypatch.setattr(io_mod.mne.io, "read_raw_brainvision", lambda path, preload=True: raw)

    result = io_mod.read_raw_preprocess(
        tmp_path / "sample.vhdr",
        montage="standard_1020",
        eog_chs=["EOG1"],
        aux_chs=["AUX"],
        reref="average",
        l_freq=0.1,
        h_freq=30.0,
        notch=[60.0],
    )

    assert result is raw
    assert raw.rename_calls == [{"FP1": "Fp1", "PZ": "Pz"}]
    assert raw.drop_calls == [["AUX"]]
    assert raw.montage_calls == [("standard_1020", "warn")]
    assert raw.reference_calls == [("average", False)]
    assert raw.notch_calls == [([60.0], 1)]
    assert raw.filter_calls == [(0.1, 30.0, 1)]
    assert "Fp1" in raw.ch_names
    assert "Pz" in raw.ch_names
    assert "AUX" not in raw.ch_names
    assert len(raw.channel_type_calls) == 2
    assert raw.channel_type_calls[0][0] == {"EOG1": "eog"}


def test_read_raw_preprocess_set_loader_supports_mastoid_reference(monkeypatch, tmp_path: Path):
    raw = DummyRaw(["TP9", "TP10", "Cz"])
    monkeypatch.setattr(io_mod.mne.io, "read_raw_eeglab", lambda path, preload=True: raw)

    result = io_mod.read_raw_preprocess(
        tmp_path / "sample.set",
        montage="standard_1020",
        eog_chs=[],
        aux_chs=[],
        reref="tp9_tp10",
        l_freq=1.0,
        h_freq=20.0,
        notch=None,
    )

    assert result is raw
    assert raw.reference_calls == [(["TP9", "TP10"], False)]
    assert raw.notch_calls == []
    assert raw.filter_calls == [(1.0, 20.0, 1)]


def test_read_raw_preprocess_rejects_missing_mastoid_channels(monkeypatch, tmp_path: Path):
    raw = DummyRaw(["Cz", "Pz"])
    monkeypatch.setattr(io_mod.mne.io, "read_raw_eeglab", lambda path, preload=True: raw)

    with pytest.raises(ValueError, match="Requested mastoid reference"):
        io_mod.read_raw_preprocess(
            tmp_path / "sample.set",
            montage="standard_1020",
            eog_chs=[],
            aux_chs=[],
            reref="mastoids",
            l_freq=1.0,
            h_freq=20.0,
            notch=None,
        )


def test_read_raw_preprocess_rejects_unknown_reference_mode(monkeypatch, tmp_path: Path):
    raw = DummyRaw(["Cz", "Pz"])
    monkeypatch.setattr(io_mod.mne.io, "read_raw_brainvision", lambda path, preload=True: raw)

    with pytest.raises(ValueError, match="Unsupported reref mode"):
        io_mod.read_raw_preprocess(
            tmp_path / "sample.vhdr",
            montage="standard_1020",
            eog_chs=[],
            aux_chs=[],
            reref="weird",
            l_freq=0.1,
            h_freq=30.0,
            notch=None,
        )


def test_read_raw_preprocess_supports_no_reference_and_missing_eog_mappings(monkeypatch, tmp_path: Path):
    raw = DummyRaw(["Cz", "Pz"])
    monkeypatch.setattr(io_mod.mne.io, "read_raw_brainvision", lambda path, preload=True: raw)

    result = io_mod.read_raw_preprocess(
        tmp_path / "sample.vhdr",
        montage="standard_1020",
        eog_chs=["Missing"],
        aux_chs=[],
        reref="none",
        l_freq=0.1,
        h_freq=30.0,
        notch=None,
    )

    assert result is raw
    assert raw.channel_type_calls == []
    assert raw.reference_calls == []


def test_events_from_annotations_positions_returns_only_event_array(monkeypatch):
    events = np.array([[1, 0, 2], [3, 0, 4]], dtype=int)
    monkeypatch.setattr(io_mod.mne, "events_from_annotations", lambda raw: (events, {"A": 2}))

    result = io_mod.events_from_annotations_positions(object())

    assert np.array_equal(result, events)


def test_parse_vmrk_markers_parses_supported_marker_lines(tmp_path: Path):
    vmrk = tmp_path / "sample.vmrk"
    vmrk.write_text(
        "\n".join(
            [
                "Brain Vision Data Exchange Marker File, Version 1.0",
                "Mk1=Stimulus,S  1,10,1,0",
                "Mk2=Stimulus,S  2,20.0,2.0,1.0",
                "Mk3=Comment,Boundary,30,1,0",
                "NotAMarker=line",
                "Mk4=BadLine,MissingFields",
            ]
        ),
        encoding="utf-8",
    )

    df = io_mod.parse_vmrk_markers(vmrk)

    assert list(df.columns) == ["mk", "mtype", "desc", "pos", "size", "chan"]
    assert len(df) == 3
    assert df.iloc[0].to_dict() == {
        "mk": 1,
        "mtype": "Stimulus",
        "desc": "S  1",
        "pos": 10,
        "size": 1,
        "chan": 0,
    }
    assert df.iloc[2]["desc"] == "Boundary"


def test_read_raw_preprocess_forwards_n_jobs_to_filters(monkeypatch, tmp_path: Path):
    """n_jobs must reach both filter calls.

    Filtering parallelizes across channels and is the main within-subject lever
    for many-channel data; dropping n_jobs here would silently leave it serial.
    """
    raw = DummyRaw(["Fz", "Cz"])
    monkeypatch.setattr(io_mod.mne.io, "read_raw_brainvision", lambda *a, **k: raw)

    path = tmp_path / "sub-01_task-oddball_eeg.vhdr"
    path.write_text("dummy", encoding="utf-8")

    io_mod.read_raw_preprocess(
        raw_path=path,
        montage="standard_1020",
        eog_chs=[],
        aux_chs=[],
        reref="none",
        l_freq=0.1,
        h_freq=30.0,
        notch=[60.0],
        n_jobs=8,
    )

    assert raw.notch_calls == [([60.0], 8)]
    assert raw.filter_calls == [(0.1, 30.0, 8)]
