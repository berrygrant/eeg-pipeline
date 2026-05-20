from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "import_manual_rejection_sets.py"
SPEC = importlib.util.spec_from_file_location("import_manual_rejection_sets", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
IMPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IMPORTER
SPEC.loader.exec_module(IMPORTER)


class RejectStruct:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_reject_mask_combines_manual_and_global(monkeypatch):
    reject = RejectStruct(
        rejmanual=np.array([0, 1, 0, 0]),
        rejglobal=np.array([0, 0, 1, 0]),
    )
    monkeypatch.setattr(IMPORTER, "_load_reject_struct", lambda _: reject)

    mask, stats = IMPORTER._reject_mask_from_set(Path("dummy.set"), 4, "manual_or_global")

    assert mask.tolist() == [False, True, True, False]
    assert stats == {
        "flagged_rejglobal": 1,
        "flagged_rejmanual": 1,
        "flagged_total": 2,
    }


def test_reject_mask_none_mode_skips_flag_lookup(monkeypatch):
    monkeypatch.setattr(IMPORTER, "_load_reject_struct", lambda _: (_ for _ in ()).throw(AssertionError("should not load")))

    mask, stats = IMPORTER._reject_mask_from_set(Path("dummy.set"), 4, "none")

    assert mask is None
    assert stats == {}


def test_reject_mask_rejects_length_mismatch(monkeypatch):
    reject = RejectStruct(rejmanual=np.array([0, 1, 0]))
    monkeypatch.setattr(IMPORTER, "_load_reject_struct", lambda _: reject)

    with pytest.raises(ValueError, match="expected 4 trials"):
        IMPORTER._reject_mask_from_set(Path("dummy.set"), 4, "manual")
