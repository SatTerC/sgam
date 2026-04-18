"""Tests for carbon balance conservation."""

import numpy as np
import pytest
from sgam.sgam import Sgam
from sgam.pft import PlantFunctionalType


def run_sgam(
    plant_type: PlantFunctionalType,
    n: int,
    gpp: np.ndarray,
    disturbances: np.ndarray,
    leaf_pool_init: float,
    stem_pool_init: float,
    root_pool_init: float,
):
    """Run SGAM with given parameters."""
    temperature = np.array([20.0] * n)
    soil_moisture = np.array([0.5] * n)
    vpd = np.array([500.0] * n)
    lue = np.array([0.5] * n)
    iwue = np.array([100.0] * n)
    week_of_year = np.arange(1, n + 1, dtype=float)

    component = Sgam(plant_type)
    result = component.forward(
        gpp=gpp,
        temperature=temperature,
        soil_moisture=soil_moisture,
        vpd=vpd,
        lue=lue,
        iwue=iwue,
        week_of_year=week_of_year,
        disturbances=disturbances,
        leaf_pool_init=leaf_pool_init,
        stem_pool_init=stem_pool_init,
        root_pool_init=root_pool_init,
    )
    return result


class TestPerTimestepConservation:
    """Test per-timestep carbon balance identity using _validate_mass_balance().

    pools[t] = pools[t-1] + npp[t] - turnover[t] - disturbance[t]
    """

    @pytest.mark.parametrize("plant_type", PlantFunctionalType)
    def test_conservation_no_disturbance(self, plant_type):
        n = 4
        gpp = np.array([5.0] * n)
        disturbances = np.zeros(n)
        leaf_init, stem_init, root_init = 1.0, 1.0, 1.0

        result = run_sgam(
            plant_type, n, gpp, disturbances, leaf_init, stem_init, root_init
        )

        assert result._validate_mass_balance()

    @pytest.mark.parametrize("plant_type", PlantFunctionalType)
    def test_conservation_with_disturbance(self, plant_type):
        n = 4
        gpp = np.array([5.0] * n)
        disturbances = np.array([0.0, 0.0, 1.0, 0.0])
        leaf_init, stem_init, root_init = 1.0, 1.0, 1.0

        result = run_sgam(
            plant_type, n, gpp, disturbances, leaf_init, stem_init, root_init
        )

        assert result._validate_mass_balance()


class TestConstantSum:
    """Test that total carbon is correctly accounted over time.

    Identity: initial_pools + sum(GPP) - sum(respiration) = final_total

    Where final_total = leaf + stem + root + litter + removed
    """

    @pytest.mark.parametrize("plant_type", PlantFunctionalType)
    def test_constant_sum_no_disturbance(self, plant_type):
        n = 30
        gpp = np.array([5.0] * n)
        disturbances = np.zeros(n)
        leaf_init, stem_init, root_init = 1.0, 1.0, 1.0

        result = run_sgam(
            plant_type, n, gpp, disturbances, leaf_init, stem_init, root_init
        )

        initial_pools = leaf_init + stem_init + root_init
        total_gpp = np.sum(gpp)
        total_resp = (
            np.sum(result.respiration.leaf)
            + np.sum(result.respiration.stem)
            + np.sum(result.respiration.root)
        )

        final_total = (
            result.pools.leaf[-1]
            + result.pools.stem[-1]
            + result.pools.root[-1]
            + result.pools.litter[-1]
            + result.pools.removed[-1]
        )

        expected = initial_pools + total_gpp - total_resp
        np.testing.assert_allclose(final_total, expected, rtol=1e-9)

    @pytest.mark.parametrize("plant_type", PlantFunctionalType)
    def test_constant_sum_with_disturbance(self, plant_type):
        n = 30
        gpp = np.array([5.0] * n)
        disturbances = np.array([0.0] * 10 + [1.0] + [0.0] * 19)
        leaf_init, stem_init, root_init = 1.0, 1.0, 1.0

        result = run_sgam(
            plant_type, n, gpp, disturbances, leaf_init, stem_init, root_init
        )

        initial_pools = leaf_init + stem_init + root_init
        total_gpp = np.sum(gpp)
        total_resp = (
            np.sum(result.respiration.leaf)
            + np.sum(result.respiration.stem)
            + np.sum(result.respiration.root)
        )

        final_total = (
            result.pools.leaf[-1]
            + result.pools.stem[-1]
            + result.pools.root[-1]
            + result.pools.litter[-1]
            + result.pools.removed[-1]
        )

        expected = initial_pools + total_gpp - total_resp
        np.testing.assert_allclose(final_total, expected, rtol=1e-9)
