import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

import eeg_pipeline.aggregate as aggregate


def _write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def _make_dataset(tmp_path: Path, subjects=("01", "02")) -> Path:
    """Build a derivatives tree with per-subject QC and metrics for each subject."""
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"
    for sub in subjects:
        eeg_dir = dataset_root / f"sub-{sub}" / "eeg"
        base = f"sub-{sub}_task-oddball"
        _write_tsv(eeg_dir / f"{base}_desc-summary_qc.tsv", [{"subject": f"sub-{sub}", "status": "OK"}])
        _write_tsv(eeg_dir / f"{base}_desc-erp_metrics.tsv", [{"subject": f"sub-{sub}", "value": 1.0}])
        _write_tsv(eeg_dir / f"{base}_desc-tfr_metrics.tsv", [{"subject": f"sub-{sub}", "power": 2.0}])
    return dataset_root


def _args() -> Namespace:
    # _save_dataframe_with_sidecar only consults args when a recording and
    # behavior source are supplied; dataset-level writes pass neither.
    return Namespace()


def test_subject_scoped_files_excludes_dataset_level_outputs(tmp_path: Path):
    dataset_root = _make_dataset(tmp_path)
    # A dataset-level table with the same desc- token, from a previous run.
    _write_tsv(dataset_root / "eeg" / "desc-erp_metrics.tsv", [{"subject": "sub-01", "value": 1.0}])

    found = aggregate._subject_scoped_files(dataset_root, aggregate.ERP_METRICS_PATTERN)

    # Only per-subject files: otherwise re-running aggregation would fold the
    # previous combined table back into its own replacement.
    assert len(found) == 2
    assert all("sub-" in p.parent.parent.name for p in found)


def test_desc_from_stem_extracts_desc_token():
    assert aggregate._desc_from_stem("sub-01_task-oddball_desc-standard_ave") == "standard"
    assert aggregate._desc_from_stem("sub-01_task-oddball_ave") is None
    assert aggregate._desc_from_stem("sub-01_desc-_ave") is None


def test_aggregate_qc_summary_combines_every_subject(tmp_path: Path):
    dataset_root = _make_dataset(tmp_path, subjects=("01", "02", "03"))

    n_files = aggregate.aggregate_qc_summary(dataset_root)

    assert n_files == 3
    combined = pd.read_csv(dataset_root / "eeg" / "desc-summary_qc.tsv", sep="\t")
    assert sorted(combined["subject"]) == ["sub-01", "sub-02", "sub-03"]
    sidecar = json.loads((dataset_root / "eeg" / "desc-summary_qc.json").read_text(encoding="utf-8"))
    assert "Description" in sidecar


def test_aggregate_metric_tables_writes_combined_erp_and_tfr(tmp_path: Path):
    dataset_root = _make_dataset(tmp_path)

    counts = aggregate.aggregate_metric_tables(dataset_root, _args())

    assert counts["erp_metrics"] == 2
    assert counts["tfr_metrics"] == 2
    assert counts["erp_timeseries"] == 0  # none written by the fixture
    erp = pd.read_csv(dataset_root / "eeg" / "desc-erp_metrics.tsv", sep="\t")
    assert sorted(erp["subject"]) == ["sub-01", "sub-02"]
    assert not (dataset_root / "eeg" / "desc-erp_timeseries.parquet").exists()


def test_aggregation_is_idempotent(tmp_path: Path, monkeypatch):
    """Re-running the gather step must not duplicate rows.

    A gather job can be retried after a partial array run, so aggregation has to
    overwrite from whatever is on disk rather than append to what it wrote before.
    """
    dataset_root = _make_dataset(tmp_path)
    monkeypatch.setattr(aggregate, "aggregate_grand_averages", lambda root, excluded=None: 0)

    aggregate.run_aggregation(dataset_root, _args())
    first = pd.read_csv(dataset_root / "eeg" / "desc-erp_metrics.tsv", sep="\t")
    aggregate.run_aggregation(dataset_root, _args())
    second = pd.read_csv(dataset_root / "eeg" / "desc-erp_metrics.tsv", sep="\t")

    assert len(first) == len(second) == 2
    pd.testing.assert_frame_equal(first, second)


def test_aggregation_picks_up_subjects_added_between_runs(tmp_path: Path, monkeypatch):
    """The gather step reflects whatever subjects exist when it runs.

    This is what lets array tasks finish independently and a single later gather
    pick all of them up.
    """
    dataset_root = _make_dataset(tmp_path, subjects=("01",))
    monkeypatch.setattr(aggregate, "aggregate_grand_averages", lambda root, excluded=None: 0)

    counts = aggregate.run_aggregation(dataset_root, _args())
    assert counts["qc"] == 1

    _make_dataset(tmp_path, subjects=("02", "03"))
    counts = aggregate.run_aggregation(dataset_root, _args())

    assert counts["qc"] == 3
    combined = pd.read_csv(dataset_root / "eeg" / "desc-summary_qc.tsv", sep="\t")
    assert sorted(combined["subject"]) == ["sub-01", "sub-02", "sub-03"]


def test_run_aggregation_reports_nothing_to_do_on_empty_tree(tmp_path: Path, capsys, monkeypatch):
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"
    dataset_root.mkdir(parents=True)
    monkeypatch.setattr(aggregate, "aggregate_grand_averages", lambda root, excluded=None: 0)

    counts = aggregate.run_aggregation(dataset_root, _args())

    assert counts["qc"] == 0
    assert "nothing to aggregate" in capsys.readouterr().out


def test_aggregate_grand_averages_groups_by_session_task_and_condition(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"
    saved: list[tuple[str, str]] = []

    class FakeEvoked:
        def __init__(self, tag):
            self.tag = tag

        def save(self, path, overwrite=True):
            saved.append((Path(path).name, self.tag))

    for sub in ("01", "02"):
        eeg_dir = dataset_root / f"sub-{sub}" / "eeg"
        eeg_dir.mkdir(parents=True)
        for cond in ("standard", "deviant"):
            path = eeg_dir / f"sub-{sub}_task-oddball_desc-{cond}_ave.fif"
            path.write_text("evoked", encoding="utf-8")
            # Sidecar carries the original capitalization of the condition.
            path.with_suffix(".json").write_text(
                json.dumps({"Condition": cond.capitalize()}), encoding="utf-8"
            )

    monkeypatch.setattr(aggregate.mne, "read_evokeds", lambda p, verbose="error": [FakeEvoked(Path(p).name)])
    monkeypatch.setattr(
        aggregate,
        "grand_averages",
        lambda evoked_map: {cond: FakeEvoked(f"ga-{cond}") for cond in evoked_map},
    )

    n_written = aggregate.aggregate_grand_averages(dataset_root)

    assert n_written == 2
    names = sorted(name for name, _ in saved)
    assert names == [
        "task-oddball_desc-grandaverage-deviant_ave.fif",
        "task-oddball_desc-grandaverage-standard_ave.fif",
    ]
    # The sidecar condition (capitalized) survives into the grand-average metadata.
    sidecar = json.loads(
        (dataset_root / "eeg" / "task-oddball_desc-grandaverage-standard_ave.json").read_text(encoding="utf-8")
    )
    assert sidecar["Condition"] == "Standard"
    assert sidecar["Task"] == "oddball"


def test_aggregate_grand_averages_skips_unreadable_evoked(tmp_path: Path, monkeypatch, capsys):
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"
    eeg_dir = dataset_root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    path = eeg_dir / "sub-01_task-oddball_desc-standard_ave.fif"
    path.write_text("corrupt", encoding="utf-8")

    def _boom(p, verbose="error"):
        raise OSError("corrupt file")

    monkeypatch.setattr(aggregate.mne, "read_evokeds", _boom)

    # One unreadable subject must not abort the whole gather.
    assert aggregate.aggregate_grand_averages(dataset_root) == 0
    assert "Could not read evoked" in capsys.readouterr().out


def test_subject_scoped_files_classifies_by_location_not_filename(tmp_path: Path):
    """A dataset-level file is excluded even if its name begins with sub-.

    Per-subject filenames also start with "sub-", so the sub- test must apply to
    the directory path, not the filename.
    """
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"
    _write_tsv(dataset_root / "eeg" / "sub-01_task-oddball_desc-erp_metrics.tsv", [{"a": 1}])
    _write_tsv(
        dataset_root / "sub-01" / "eeg" / "sub-01_task-oddball_desc-erp_metrics.tsv",
        [{"a": 1}],
    )

    found = aggregate._subject_scoped_files(dataset_root, aggregate.ERP_METRICS_PATTERN)

    assert [p.parent.parent.name for p in found] == ["sub-01"]



def test_aggregate_preserves_zero_padded_entity_labels(tmp_path: Path):
    """BIDS entity labels must survive the TSV round-trip as written.

    pandas infers a zero-padded run like "01" as the integer 1, which would make
    the combined tables disagree with both the filenames and the per-subject
    tables they were built from.
    """
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"
    _write_tsv(
        dataset_root / "sub-01" / "eeg" / "sub-01_task-oddball_run-01_desc-summary_qc.tsv",
        [{"subject": "sub-01", "session": "", "task": "oddball", "run": "01", "n_epochs": 40}],
    )

    aggregate.aggregate_qc_summary(dataset_root)

    combined = pd.read_csv(
        dataset_root / "eeg" / "desc-summary_qc.tsv", sep="\t", dtype={"run": str}
    )
    assert combined.loc[0, "run"] == "01"
    # Genuinely numeric columns must still be numeric.
    assert int(combined.loc[0, "n_epochs"]) == 40


def test_aggregate_grand_averages_merges_conditions_differing_only_by_case(tmp_path: Path, monkeypatch):
    """Condition labels differing only in case must form ONE grand average.

    The output path lower-cases the condition, so grouping by raw label would let
    "Standard" and "standard" produce two groups writing to the same path -- one
    silently overwriting the other, each built from part of the cohort.
    """
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"
    group_sizes: list[int] = []

    class FakeEvoked:
        def save(self, path, overwrite=True):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("ga", encoding="utf-8")

    # sub-01 carries a sidecar (original case); sub-02 has none, so its condition
    # falls back to the lower-cased desc- token in the filename.
    for sub, with_sidecar in (("01", True), ("02", False)):
        eeg_dir = dataset_root / f"sub-{sub}" / "eeg"
        eeg_dir.mkdir(parents=True)
        path = eeg_dir / f"sub-{sub}_task-oddball_desc-standard_ave.fif"
        path.write_text("evoked", encoding="utf-8")
        if with_sidecar:
            path.with_suffix(".json").write_text(json.dumps({"Condition": "Standard"}), encoding="utf-8")

    monkeypatch.setattr(aggregate.mne, "read_evokeds", lambda p, verbose="error": [FakeEvoked()])

    def _grand_averages(evoked_map):
        for evokeds in evoked_map.values():
            group_sizes.append(len(evokeds))
        return {cond: FakeEvoked() for cond in evoked_map}

    monkeypatch.setattr(aggregate, "grand_averages", _grand_averages)

    n_written = aggregate.aggregate_grand_averages(dataset_root)

    assert n_written == 1
    assert group_sizes == [2]  # both subjects in one group, not 1+1
    sidecar = json.loads(
        (dataset_root / "eeg" / "task-oddball_desc-grandaverage-standard_ave.json").read_text(encoding="utf-8")
    )
    # Original capitalization is retained for metadata.
    assert sidecar["Condition"] == "Standard"


def test_aggregation_excludes_recordings_the_latest_run_skipped(tmp_path: Path, monkeypatch):
    """Stale metrics from an earlier run must not survive a later skip.

    Aggregation reads whatever per-subject files are on disk, and a skipped
    recording's metrics/evokeds from a previous run are not deleted. Re-running
    with a stricter setting (a tighter --max_reject_rate, say) rewrites the QC row
    to a skip status; without filtering on that status the stale metrics would be
    folded back in, so the dataset would report a subject excluded while still
    averaging it into the combined tables and grand averages.
    """
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"
    for sub, status in (("01", "OK"), ("02", "SKIP_REJECT_RATE")):
        eeg_dir = dataset_root / f"sub-{sub}" / "eeg"
        base = f"sub-{sub}_task-oddball_run-01"
        _write_tsv(
            eeg_dir / f"{base}_desc-summary_qc.tsv",
            [{"subject": f"sub-{sub}", "session": "", "task": "oddball", "run": "01", "status": status}],
        )
        # BOTH subjects have metrics on disk: sub-02's are stale, left by the
        # earlier run that had not yet excluded it.
        _write_tsv(eeg_dir / f"{base}_desc-erp_metrics.tsv", [{"subject": f"sub-{sub}", "value": 1.0}])

    monkeypatch.setattr(aggregate, "aggregate_grand_averages", lambda root, excluded=None: 0)
    aggregate.run_aggregation(dataset_root, _args())

    combined = pd.read_csv(dataset_root / "eeg" / "desc-erp_metrics.tsv", sep="\t")
    assert sorted(combined["subject"]) == ["sub-01"]
    # The QC summary still reports both, including the skip -- that is the record.
    qc = pd.read_csv(dataset_root / "eeg" / "desc-summary_qc.tsv", sep="\t")
    assert sorted(qc["subject"]) == ["sub-01", "sub-02"]


def test_aggregation_excludes_skipped_recordings_from_grand_averages(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"
    averaged: list[int] = []

    class FakeEvoked:
        def save(self, path, overwrite=True):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("ga", encoding="utf-8")

    for sub, status in (("01", "OK"), ("02", "SKIP_REJECT_RATE")):
        eeg_dir = dataset_root / f"sub-{sub}" / "eeg"
        eeg_dir.mkdir(parents=True)
        base = f"sub-{sub}_task-oddball_run-01"
        _write_tsv(
            eeg_dir / f"{base}_desc-summary_qc.tsv",
            [{"subject": f"sub-{sub}", "session": "", "task": "oddball", "run": "01", "status": status}],
        )
        path = eeg_dir / f"{base}_desc-standard_ave.fif"
        path.write_text("evoked", encoding="utf-8")
        path.with_suffix(".json").write_text(json.dumps({"Condition": "Standard"}), encoding="utf-8")

    monkeypatch.setattr(aggregate.mne, "read_evokeds", lambda p, verbose="error": [FakeEvoked()])

    def _grand_averages(evoked_map):
        for evokeds in evoked_map.values():
            averaged.append(len(evokeds))
        return {cond: FakeEvoked() for cond in evoked_map}

    monkeypatch.setattr(aggregate, "grand_averages", _grand_averages)

    aggregate.aggregate_grand_averages(dataset_root, aggregate._excluded_recording_keys(dataset_root))

    # Only sub-01 contributes; the skipped sub-02's stale evoked is left out.
    assert averaged == [1]


def test_is_excluded_keeps_unparseable_filenames(tmp_path: Path):
    # No key means no evidence the recording was skipped; dropping it would
    # silently shrink the outputs.
    assert aggregate._is_excluded(Path("not-a-bids-name.tsv"), {("01", "", "oddball", "01")}) is False
    assert aggregate._is_excluded(Path("sub-01_task-oddball_x.tsv"), None) is False


def test_aggregation_removes_dataset_tables_that_have_no_inputs(tmp_path: Path, monkeypatch):
    """A gather with no inputs must not leave the previous table looking current.

    Aggregation only writes outputs it can reconstruct, so a stale dataset table
    would otherwise survive a gather that found nothing to build it from, and
    --plot_figures would consume it as belonging to this run.
    """
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"
    stale = dataset_root / "eeg" / "desc-erp_metrics.tsv"
    _write_tsv(stale, [{"subject": "sub-99", "value": 1.0}])
    stale.with_suffix(".json").write_text("{}", encoding="utf-8")
    # A QC input exists so aggregation runs, but no ERP metrics files do.
    _write_tsv(
        dataset_root / "sub-01" / "eeg" / "sub-01_task-oddball_desc-summary_qc.tsv",
        [{"subject": "sub-01", "session": "", "task": "oddball", "run": "", "status": "OK"}],
    )

    monkeypatch.setattr(aggregate, "aggregate_grand_averages", lambda root, excluded=None: 0)
    aggregate.run_aggregation(dataset_root, _args())

    assert not stale.exists()
    assert not stale.with_suffix(".json").exists()


def test_aggregation_removes_grand_averages_for_vanished_conditions(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"

    class FakeEvoked:
        def save(self, path, overwrite=True):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("ga", encoding="utf-8")

    # A grand average for a condition that no longer has any subject evokeds.
    stale = dataset_root / "eeg" / "task-oddball_desc-grandaverage-deviant_ave.fif"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")

    eeg_dir = dataset_root / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    path = eeg_dir / "sub-01_task-oddball_desc-standard_ave.fif"
    path.write_text("evoked", encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps({"Condition": "Standard"}), encoding="utf-8")

    monkeypatch.setattr(aggregate.mne, "read_evokeds", lambda p, verbose="error": [FakeEvoked()])
    monkeypatch.setattr(aggregate, "grand_averages", lambda m: {c: FakeEvoked() for c in m})

    aggregate.aggregate_grand_averages(dataset_root)

    assert not stale.exists()
    assert (dataset_root / "eeg" / "task-oddball_desc-grandaverage-standard_ave.fif").exists()
    # Per-subject data must never be touched by dataset-level cleanup.
    assert path.exists()


def test_dataset_scoped_files_never_reaches_subject_data(tmp_path: Path):
    dataset_root = tmp_path / "derivatives" / "eeg-pipeline"
    (dataset_root / "eeg").mkdir(parents=True)
    (dataset_root / "sub-01" / "eeg").mkdir(parents=True)
    ds = dataset_root / "eeg" / "task-oddball_desc-grandaverage-standard_ave.fif"
    ds.write_text("ga", encoding="utf-8")
    (dataset_root / "sub-01" / "eeg" / "sub-01_task-oddball_desc-standard_ave.fif").write_text("s", encoding="utf-8")

    assert aggregate._dataset_scoped_files(dataset_root, "*_ave.fif") == [ds]
