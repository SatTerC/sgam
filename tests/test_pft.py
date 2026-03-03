import pytest
from sgam.pft import PlantFunctionalType, get_default_pft_params


class TestPFTParameters:
    @pytest.mark.parametrize("pft", PlantFunctionalType)
    def test_allocation_sums_to_one(self, pft):
        params = get_default_pft_params(pft)
        total = (
            params.leaf_base_allocation
            + params.stem_base_allocation
            + params.root_base_allocation
        )
        assert abs(total - 1.0) < 1e-10

    @pytest.mark.parametrize("pft", PlantFunctionalType)
    def test_turnover_rates_in_valid_range(self, pft):
        params = get_default_pft_params(pft)
        assert 0.0 < params.leaf_turnover_rate <= 1.0
        assert 0.0 <= params.stem_turnover_rate <= 1.0
        assert 0.0 < params.root_turnover_rate <= 1.0

    @pytest.mark.parametrize("pft", PlantFunctionalType)
    def test_leaf_carbon_area_positive(self, pft):
        params = get_default_pft_params(pft)
        assert params.leaf_carbon_area > 0.0

    @pytest.mark.parametrize("pft", PlantFunctionalType)
    def test_disturbance_limit_in_valid_range(self, pft):
        params = get_default_pft_params(pft)
        assert 0.0 < params.disturbance_limit <= 1.0
