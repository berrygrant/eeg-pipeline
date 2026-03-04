import numpy as np
import pytest

from eeg_pipeline.align import (
    align_marker_positions_to_codes,
    keep_best_dense_markers_to_count,
    keep_by_gap_heuristic,
    marker_gap_stats,
)


def test_marker_gap_stats_returns_summary_for_short_inputs():
    assert marker_gap_stats(np.array([], dtype=int), sfreq=100.0) == {"n": 0}
    assert marker_gap_stats(np.array([25], dtype=int), sfreq=100.0) == {"n": 1}


def test_marker_gap_stats_computes_gap_quantiles():
    stats = marker_gap_stats(np.array([0, 100, 250, 400]), sfreq=100.0)

    assert stats["n"] == 4
    assert stats["dt_min"] == pytest.approx(1.0)
    assert stats["dt_p50"] == pytest.approx(1.5)
    assert stats["dt_max"] == pytest.approx(1.5)


def test_keep_by_gap_heuristic_keeps_dense_interior_markers():
    markers = np.array([0, 10, 20, 30], dtype=int)

    kept = keep_by_gap_heuristic(markers, sfreq=10.0, gap_s=1.1)

    assert np.array_equal(kept, np.array([1, 2], dtype=int))


def test_keep_by_gap_heuristic_handles_empty_inputs():
    kept = keep_by_gap_heuristic(np.array([], dtype=int), sfreq=10.0, gap_s=1.0)

    assert np.array_equal(kept, np.array([], dtype=int))


def test_keep_best_dense_markers_to_count_returns_dense_cluster():
    markers = np.array([0, 10, 20, 30, 100], dtype=int)

    kept = keep_best_dense_markers_to_count(markers, sfreq=10.0, target_n=3)

    assert np.array_equal(kept, np.array([1, 2, 3], dtype=int))


def test_keep_best_dense_markers_to_count_returns_all_when_target_matches():
    markers = np.array([0, 10, 20], dtype=int)

    kept = keep_best_dense_markers_to_count(markers, sfreq=10.0, target_n=3)

    assert np.array_equal(kept, np.array([0, 1, 2], dtype=int))


def test_keep_best_dense_markers_to_count_rejects_large_target():
    with pytest.raises(ValueError, match="target_n"):
        keep_best_dense_markers_to_count(np.array([0, 10], dtype=int), sfreq=10.0, target_n=3)


def test_align_marker_positions_to_codes_auto_drops_to_match_length():
    markers = np.array([0, 10, 20, 30], dtype=int)
    codes = np.array([101, 102], dtype=int)

    aligned, diag = align_marker_positions_to_codes(
        markers_pos=markers,
        sfreq=10.0,
        codes=codes,
        gap_s=None,
        auto_drop_to_count=True,
    )

    assert np.array_equal(aligned, np.array([10, 20], dtype=int))
    assert diag == {
        "markers_original": 4,
        "markers_dropped_by_gap": 0,
        "markers_dropped_by_auto": 2,
    }


def test_align_marker_positions_to_codes_errors_when_counts_still_mismatch():
    with pytest.raises(ValueError, match="Alignment failed"):
        align_marker_positions_to_codes(
            markers_pos=np.array([0, 10, 20], dtype=int),
            sfreq=10.0,
            codes=np.array([101, 102], dtype=int),
            gap_s=None,
            auto_drop_to_count=False,
        )


def test_align_marker_positions_to_codes_can_match_exactly_after_gap_filter():
    aligned, diag = align_marker_positions_to_codes(
        markers_pos=np.array([0, 10, 20, 30], dtype=int),
        sfreq=10.0,
        codes=np.array([101, 102], dtype=int),
        gap_s=1.1,
        auto_drop_to_count=False,
    )

    assert np.array_equal(aligned, np.array([10, 20], dtype=int))
    assert diag == {
        "markers_original": 4,
        "markers_dropped_by_gap": 2,
        "markers_dropped_by_auto": 0,
    }
