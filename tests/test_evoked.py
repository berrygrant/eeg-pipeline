from eeg_pipeline.evoked import compute_evokeds, grand_averages


def test_compute_evokeds_skips_missing_conditions(synthetic_epochs):
    evokeds = compute_evokeds(synthetic_epochs, ["Standard", "Missing"])

    assert list(evokeds) == ["Standard"]
    assert evokeds["Standard"].comment == "Standard"


def test_grand_averages_skips_empty_condition_lists(synthetic_epochs):
    standard = synthetic_epochs["Standard"].average()
    deviant = synthetic_epochs["Deviant"].average()

    ga = grand_averages({"Standard": [standard, standard], "Deviant": [deviant, deviant], "Empty": []})

    assert set(ga) == {"Standard", "Deviant"}
    assert ga["Standard"].data.shape == standard.data.shape
