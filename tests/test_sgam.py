import numpy as np
from sgam.sgam import SgamComponent
from sgam.pft import PlantFunctionalType, PftParams, get_default_pft_params


class TestComputeCue:
    def test_cue_output_range(self):
        component = SgamComponent(PlantFunctionalType.TREE)
        lue = np.array([0.1, 0.5, 0.9])
        iwue = np.array([50.0, 100.0, 200.0])
        cue = component.compute_cue(lue, iwue)
        assert np.all(cue >= 0.2)
        assert np.all(cue <= 0.9)


class TestComputeDroughtModifier:
    def test_no_stress_returns_one(self):
        component = SgamComponent(PlantFunctionalType.TREE)
        soil_moisture = np.array([0.5, 0.6, 0.7])
        vpd = np.array([200.0, 300.0, 400.0])
        modifier = component.compute_drought_modifier(soil_moisture, vpd)
        assert np.all(modifier >= 0.9)

    def test_moisture_stress_decreases_modifier(self):
        component = SgamComponent(PlantFunctionalType.TREE)
        soil_moisture = np.array([0.05])
        vpd = np.array([500.0])
        modifier = component.compute_drought_modifier(soil_moisture, vpd)
        assert modifier[0] < 1.0


class TestComputeAllocationFractions:
    def test_allocations_sum_to_one(self):
        component = SgamComponent(PlantFunctionalType.TREE)
        temperature = np.array([20.0, 25.0])
        soil_moisture = np.array([0.5, 0.5])
        vpd = np.array([500.0, 500.0])
        week_of_year = np.array([1.0, 2.0])
        leaf, stem, root = component.compute_allocation_fractions(
            temperature,
            soil_moisture,
            vpd,
            week_of_year,
        )
        total = leaf + stem + root
        np.testing.assert_allclose(total, np.ones(2), rtol=1e-10)


class TestForwardCropDisturbance:
    def test_crop_resets_on_disturbance(self):
        component = SgamComponent(PlantFunctionalType.CROP)

        n = 10
        temperature = np.array([15.0] * n)
        soil_moisture = np.array([0.5] * n)
        vpd = np.array([500.0] * n)
        gpp = np.array([10.0, 10.0, 10.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        iwue = np.array([100.0] * n)
        lue = np.array([0.5] * n)
        week_of_year = np.arange(1, n + 1, dtype=float)
        disturbances = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = component.forward(
            gpp,
            temperature,
            soil_moisture,
            vpd,
            lue,
            iwue,
            week_of_year,
            disturbances,
            leaf_pool_init=5.0,
            stem_pool_init=10.0,
            root_pool_init=5.0,
        )

        assert result["leaf_pool"][3] == 0.0
        assert result["removed"][3] > 0.0


class TestForwardTreeDisturbance:
    def test_tree_loses_partial_carbon_on_disturbance(self):
        component = SgamComponent(PlantFunctionalType.TREE)

        n = 10
        temperature = np.array([15.0] * n)
        soil_moisture = np.array([0.5] * n)
        vpd = np.array([500.0] * n)
        gpp = np.array([10.0, 10.0, 10.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        iwue = np.array([100.0] * n)
        lue = np.array([0.5] * n)
        week_of_year = np.arange(1, n + 1, dtype=float)
        disturbances = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = component.forward(
            gpp,
            temperature,
            soil_moisture,
            vpd,
            lue,
            iwue,
            week_of_year,
            disturbances,
            leaf_pool_init=5.0,
            stem_pool_init=10.0,
            root_pool_init=5.0,
        )

        assert result["leaf_pool"][3] < 5.0
        assert result["leaf_pool"][3] > 0.0


class TestForwardIntegration:
    def test_physical_sanity_check(self):
        component = SgamComponent(PlantFunctionalType.GRASS)

        n = 30
        temperature = np.array([20.0] * n)
        soil_moisture = np.array([0.5] * n)
        vpd = np.array([500.0] * n)
        gpp = np.array([5.0] * n)
        iwue = np.array([100.0] * n)
        lue = np.array([0.5] * n)
        week_of_year = np.arange(1, n + 1, dtype=float)
        disturbances = np.zeros(n)

        result = component.forward(
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

        assert np.all(result["leaf_pool"] >= 0.0)
        assert np.all(result["stem_pool"] >= 0.0)
        assert np.all(result["root_pool"] >= 0.0)


class TestCustomPftParams:
    def test_custom_pft_params_overrides_defaults(self):
        custom_params = PftParams(
            leaf_base_allocation=0.3,
            stem_base_allocation=0.5,
            root_base_allocation=0.2,
            leaf_turnover_rate=0.01,
            stem_turnover_rate=0.001,
            root_turnover_rate=0.02,
            lue_max=2.0,
            iwue_max=400.0,
            disturbance_threshold=0.25,
            disturbance_leaf_loss_frac=0.5,
            leaf_carbon_area=50.0,
            wilting_point=0.10,
            field_capacity=0.30,
            vpd_threshold=700.0,
            vpd_sensitivity=0.0006,
        )
        component = SgamComponent(PlantFunctionalType.TREE, pft_params=custom_params)
        assert component.pft_params is custom_params
        assert component.pft_params.lue_max == 2.0
        assert component.pft_params.leaf_base_allocation == 0.3

    def test_default_params_when_none(self):
        component = SgamComponent(PlantFunctionalType.GRASS)
        defaults = get_default_pft_params(PlantFunctionalType.GRASS)
        assert component.pft_params is defaults


class TestDynamicAllocationToggle:
    def test_dynamic_allocation_default_true(self):
        component = SgamComponent(PlantFunctionalType.TREE)
        assert component.use_dynamic_allocation is True

    def test_static_allocation_mode(self):
        component = SgamComponent(
            PlantFunctionalType.TREE, use_dynamic_allocation=False
        )
        assert component.use_dynamic_allocation is False

        temperature = np.array([20.0, 30.0])
        soil_moisture = np.array([0.5, 0.5])
        vpd = np.array([500.0, 500.0])
        week_of_year = np.array([1.0, 26.0])

        leaf, stem, root = component.compute_allocation_fractions(
            temperature,
            soil_moisture,
            vpd,
            week_of_year,
        )

        assert np.all(leaf == component.pft_params.leaf_base_allocation)
        assert np.all(stem == component.pft_params.stem_base_allocation)
        assert np.all(root == component.pft_params.root_base_allocation)

    def test_dynamic_allocation_varies_with_environment(self):
        component = SgamComponent(PlantFunctionalType.TREE, use_dynamic_allocation=True)

        temperature = np.array([20.0, 30.0])
        soil_moisture = np.array([0.5, 0.5])
        vpd = np.array([500.0, 500.0])
        week_of_year = np.array([1.0, 26.0])

        leaf, stem, root = component.compute_allocation_fractions(
            temperature,
            soil_moisture,
            vpd,
            week_of_year,
        )

        total = leaf + stem + root
        np.testing.assert_allclose(total, np.ones(2), rtol=1e-10)
        assert not np.allclose(leaf[0], leaf[1], rtol=1e-10)

    def test_override_dynamic_allocation_at_call_time(self):
        component = SgamComponent(
            PlantFunctionalType.TREE, use_dynamic_allocation=False
        )

        temperature = np.array([20.0, 30.0])
        soil_moisture = np.array([0.5, 0.5])
        vpd = np.array([500.0, 500.0])
        week_of_year = np.array([1.0, 26.0])

        leaf_static, _, _ = component.compute_allocation_fractions(
            temperature, soil_moisture, vpd, week_of_year, use_dynamic_allocation=False
        )

        leaf_dynamic, _, _ = component.compute_allocation_fractions(
            temperature, soil_moisture, vpd, week_of_year, use_dynamic_allocation=True
        )

        assert np.all(leaf_static == component.pft_params.leaf_base_allocation)
        assert not np.allclose(leaf_static, leaf_dynamic, rtol=1e-10)
