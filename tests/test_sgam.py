import numpy as np
from sgam.sgam import SgamComponent
from sgam.pft import PlantFunctionalType


class TestComputeCue:
    def test_cue_output_range(self):
        component = SgamComponent(PlantFunctionalType.TREE)
        lue = np.array([0.1, 0.5, 0.9])
        iwue = np.array([50.0, 100.0, 200.0])
        cue = component.compute_cue(lue, iwue)
        assert np.all(cue >= 0.2)
        assert np.all(cue <= 0.9)


class TestComputeDroughtModifier:
    def test_no_stress_returns_zero(self):
        component = SgamComponent(PlantFunctionalType.TREE)
        soil_moisture = np.array([1.0, 1.0, 1.0])
        vpd = np.array([0.0, 0.0, 0.0])
        modifier = component.compute_drought_modifier(
            soil_moisture, vpd, moisture_threshold=0.5, vpd_max=1000.0
        )
        np.testing.assert_array_equal(modifier, np.zeros(3))

    def test_moisture_stress_increases_root_allocation(self):
        component = SgamComponent(PlantFunctionalType.TREE)
        soil_moisture = np.array([0.1])
        vpd = np.array([0.0])
        modifier = component.compute_drought_modifier(
            soil_moisture, vpd, moisture_threshold=0.5, vpd_max=1000.0
        )
        assert modifier[0] > 0.0


class TestComputeAllocationPercentages:
    def test_allocations_sum_to_one(self):
        component = SgamComponent(
            PlantFunctionalType.TREE,
            leaf_base_allocation=0.3,
            stem_base_allocation=0.4,
            root_base_allocation=0.3,
        )
        temperature = np.array([20.0, 25.0])
        day_of_year = np.array([1.0, 2.0])
        soil_moisture = np.array([0.5, 0.5])
        vpd = np.array([500.0, 500.0])
        leaf, stem, root = component.compute_allocation_percentages(
            temperature,
            day_of_year,
            soil_moisture,
            vpd,
            moisture_threshold=0.5,
            vpd_max=1000.0,
        )
        total = leaf + stem + root
        np.testing.assert_allclose(total, np.ones(2), rtol=1e-10)


class TestForwardCropDisturbance:
    def test_crop_resets_on_disturbance(self):
        component = SgamComponent(PlantFunctionalType.CROP)

        n = 10
        temperature = np.array([15.0] * n)
        vpd = np.array([500.0] * n)
        lai_obs = np.array([2.0, 2.0, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        day_of_year = np.arange(1, n + 1, dtype=float)
        soil_moisture = np.array([0.5] * n)
        gpp = np.array([10.0, 10.0, 10.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        iwue = np.array([100.0] * n)
        lue = np.array([0.5] * n)

        result = component.forward(
            temperature,
            vpd,
            lai_obs,
            day_of_year,
            soil_moisture,
            gpp,
            iwue,
            lue,
            leaf_pool_init=5.0,
            stem_pool_init=10.0,
            root_pool_init=5.0,
        )

        assert result["leaf_pool_size"][3] == 0.0
        assert result["disturbance_losses"][3] > 0.0


class TestForwardTreeDisturbance:
    def test_tree_loses_partial_carbon_on_disturbance(self):
        component = SgamComponent(PlantFunctionalType.TREE)

        n = 10
        temperature = np.array([15.0] * n)
        vpd = np.array([500.0] * n)
        lai_obs = np.array([2.0, 2.0, 2.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        day_of_year = np.arange(1, n + 1, dtype=float)
        soil_moisture = np.array([0.5] * n)
        gpp = np.array([10.0, 10.0, 10.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        iwue = np.array([100.0] * n)
        lue = np.array([0.5] * n)

        result = component.forward(
            temperature,
            vpd,
            lai_obs,
            day_of_year,
            soil_moisture,
            gpp,
            iwue,
            lue,
            leaf_pool_init=5.0,
            stem_pool_init=10.0,
            root_pool_init=5.0,
        )

        assert result["leaf_pool_size"][3] < 5.0
        assert result["leaf_pool_size"][3] > 0.0


class TestForwardIntegration:
    def test_physical_sanity_check(self):
        component = SgamComponent(PlantFunctionalType.GRASS)

        n = 30
        temperature = np.array([20.0] * n)
        vpd = np.array([500.0] * n)
        lai_obs = np.array([1.0] * n)
        day_of_year = np.arange(1, n + 1, dtype=float)
        soil_moisture = np.array([0.5] * n)
        gpp = np.array([5.0] * n)
        iwue = np.array([100.0] * n)
        lue = np.array([0.5] * n)

        result = component.forward(
            temperature,
            vpd,
            lai_obs,
            day_of_year,
            soil_moisture,
            gpp,
            iwue,
            lue,
            leaf_pool_init=1.0,
            stem_pool_init=1.0,
            root_pool_init=1.0,
        )

        assert np.all(result["leaf_pool_size"] >= 0.0)
        assert np.all(result["stem_pool_size"] >= 0.0)
        assert np.all(result["root_pool_size"] >= 0.0)
        assert np.all(result["leaf_area_index"] >= 0.0)
        np.testing.assert_allclose(
            result["leaf_area_index"],
            result["leaf_pool_size"] / component.leaf_carbon_area,
            rtol=1e-10,
        )
