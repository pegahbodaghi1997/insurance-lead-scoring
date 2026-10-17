import numpy as np
import pandas as pd

from leadscore.monitoring import psi
from leadscore.predict import assign_priority

TIERS = {"P1": 0.10, "P2": 0.30}


def _scores(n=100):
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.5, n)
    margin = rng.uniform(1e5, 1e6, n)
    return pd.DataFrame({"lead_id": [f"L{i}" for i in range(n)], "probability": p,
                         "expected_value": p * margin})


def test_rank_by_probability_and_tiers():
    out = assign_priority(_scores(), "probability", TIERS)

    assert sorted(out.priority_rank) == list(range(1, 101))
    top = out.sort_values("priority_rank")
    assert top.probability.is_monotonic_decreasing
    assert out.priority_tier.value_counts().to_dict() == {"P1": 10, "P2": 20, "P3": 70}


def test_rank_by_expected_value():
    out = assign_priority(_scores(), "expected_value", TIERS)
    first = out.loc[out.priority_rank == 1].iloc[0]
    assert first.expected_value == out.expected_value.max()


def test_psi_is_zero_for_same_distribution_and_grows_with_shift():
    same = psi([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25])
    shifted = psi([0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4])
    assert abs(same) < 1e-9
    assert shifted > 0.1
