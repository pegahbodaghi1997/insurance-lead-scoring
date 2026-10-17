from leadscore.train import decide_incoming_call

WITHOUT = {"roc_auc": 0.70, "pr_auc": 0.200}


def test_small_gain_is_excluded():
    d = decide_incoming_call(WITHOUT, {"roc_auc": 0.701, "pr_auc": 0.202}, False, 0.005)
    assert not d["include"]
    assert d["chosen_variant"] == "without_call"
    assert "negligible" in d["reason"]


def test_large_gain_without_confirmed_timing_is_still_excluded():
    d = decide_incoming_call(WITHOUT, {"roc_auc": 0.75, "pr_auc": 0.260}, False, 0.005)
    assert not d["include"]
    assert "cannot confirm" in d["reason"]


def test_confirmed_timing_and_real_gain_is_included():
    d = decide_incoming_call(WITHOUT, {"roc_auc": 0.75, "pr_auc": 0.260}, True, 0.005)
    assert d["include"]
    assert d["chosen_variant"] == "with_call"
    assert d["pr_auc_gain"] == 0.06
