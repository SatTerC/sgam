"""Property-based tests using Hypothesis.

These complement the hand-crafted unit tests by fuzzing inputs.
Run with: pytest tests/test_properties.py
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from sgam.pft import PlantFunctionalType
from sgam.sgam import Sgam


@given(
    lue=arrays(np.float64, 20, elements=st.floats(0.0, 10.0, allow_nan=False)),
    iwue=arrays(np.float64, 20, elements=st.floats(0.0, 1000.0, allow_nan=False)),
    pft=st.sampled_from(list(PlantFunctionalType)),
)
def test_cue_always_in_valid_range(lue, iwue, pft):
    """CUE must always lie in [0.2, 0.7] for any non-negative inputs."""
    cue, _, _ = Sgam(pft).compute_cue(lue, iwue)
    assert np.all(cue >= 0.2)
    assert np.all(cue <= 0.7)


@given(
    soil_moisture=arrays(np.float64, 10, elements=st.floats(0.0, 1.0, allow_nan=False)),
    vpd=arrays(np.float64, 10, elements=st.floats(0.0, 5000.0, allow_nan=False)),
    pft=st.sampled_from(list(PlantFunctionalType)),
)
def test_drought_modifier_always_in_0_1(soil_moisture, vpd, pft):
    modifier = Sgam(pft).compute_drought_modifier(soil_moisture, vpd)
    assert np.all(modifier >= 0.0)
    assert np.all(modifier <= 1.0)


@given(
    temperature=arrays(
        np.float64, 10, elements=st.floats(-20.0, 45.0, allow_nan=False)
    ),
    soil_moisture=arrays(np.float64, 10, elements=st.floats(0.0, 1.0, allow_nan=False)),
    vpd=arrays(np.float64, 10, elements=st.floats(0.0, 5000.0, allow_nan=False)),
    week_of_year=arrays(np.float64, 10, elements=st.floats(1.0, 52.0, allow_nan=False)),
    pft=st.sampled_from(list(PlantFunctionalType)),
)
def test_allocation_fractions_sum_to_one(
    temperature, soil_moisture, vpd, week_of_year, pft
):
    leaf, stem, root = Sgam(pft).compute_allocation_fractions(
        temperature, soil_moisture, vpd, week_of_year
    )
    np.testing.assert_allclose(leaf + stem + root, np.ones(10), atol=1e-12)


@given(
    n=st.integers(min_value=2, max_value=52),
    gpp_val=st.floats(0.1, 20.0, allow_nan=False),
    pft=st.sampled_from(list(PlantFunctionalType)),
)
@settings(max_examples=50)
def test_mass_balance_holds_for_random_inputs(n, gpp_val, pft):
    """Mass balance must hold for any combination of PFT and constant GPP."""
    gpp = np.full(n, gpp_val)
    temperature = np.full(n, 20.0)
    soil_moisture = np.full(n, 0.4)
    vpd = np.full(n, 600.0)
    lue = np.full(n, 0.5)
    iwue = np.full(n, 100.0)
    week_of_year = np.arange(1, n + 1, dtype=float)
    disturbances = np.zeros(n)

    result = Sgam(pft).forward(
        gpp,
        temperature,
        soil_moisture,
        vpd,
        lue,
        iwue,
        week_of_year,
        disturbances,
        leaf_pool_init=1.0,
        stem_pool_init=1.0,
        root_pool_init=1.0,
    )
    assert result._validate_mass_balance()


@given(
    severity=st.floats(0.01, 1.0, allow_nan=False),
    pft=st.sampled_from(
        [
            PlantFunctionalType.GRASS,
            PlantFunctionalType.SHRUB,
            PlantFunctionalType.TREE,
        ]
    ),
)
def test_non_crop_disturbance_never_creates_negative_pools(severity, pft):
    """Leaf loss from a disturbance must not make any pool negative."""
    n = 5
    gpp = np.array([5.0] * n)
    disturbances = np.array([0.0, 0.0, severity, 0.0, 0.0])
    temperature = np.full(n, 20.0)
    soil_moisture = np.full(n, 0.4)
    vpd = np.full(n, 600.0)
    lue = np.full(n, 0.5)
    iwue = np.full(n, 100.0)
    week_of_year = np.arange(1, n + 1, dtype=float)

    result = Sgam(pft).forward(
        gpp,
        temperature,
        soil_moisture,
        vpd,
        lue,
        iwue,
        week_of_year,
        disturbances,
        leaf_pool_init=1.0,
        stem_pool_init=1.0,
        root_pool_init=1.0,
    )
    assert np.all(result.pools.leaf >= 0.0)
    assert np.all(result.pools.stem >= 0.0)
    assert np.all(result.pools.root >= 0.0)
