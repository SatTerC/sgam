"""Tests for disturbance detection in SgamComponent."""

import numpy as np
import pytest
from sgam.pft import PlantFunctionalType
from sgam.sgam import SgamComponent


class TestDisturbanceDetection:
    """Test suite for disturbance event detection.

    Tests both crop (which resets pools completely) and non-crop PFTs
    (which lose a fraction of biomass).
    """

    @pytest.mark.parametrize(
        "pft", [PlantFunctionalType.CROP, PlantFunctionalType.GRASS]
    )
    def test_zero_disturbance_events(
        self,
        pft,
        driving_data_no_disturbance,
        initial_pools,
    ):
        """Test that no disturbance events are detected when driving data is constant."""
        component = SgamComponent(pft)
        result = component.forward(
            **driving_data_no_disturbance,
            **initial_pools,
        )
        disturbance_count = np.count_nonzero(result["disturbance"])
        assert disturbance_count == 0

    @pytest.mark.parametrize(
        "pft", [PlantFunctionalType.CROP, PlantFunctionalType.GRASS]
    )
    def test_one_disturbance_event(
        self,
        pft,
        driving_data_one_disturbance,
        initial_pools,
    ):
        """Test that exactly one disturbance event is detected."""
        component = SgamComponent(pft)
        result = component.forward(
            **driving_data_one_disturbance,
            **initial_pools,
        )
        disturbance_count = np.count_nonzero(result["disturbance"])
        assert disturbance_count == 1

    @pytest.mark.parametrize(
        "pft", [PlantFunctionalType.CROP, PlantFunctionalType.GRASS]
    )
    def test_two_disturbance_events(
        self,
        pft,
        driving_data_two_disturbances,
        initial_pools,
    ):
        """Test that exactly two disturbance events are detected."""
        component = SgamComponent(pft)
        result = component.forward(
            **driving_data_two_disturbances,
            **initial_pools,
        )
        disturbance_count = np.count_nonzero(result["disturbance"])
        assert disturbance_count == 2
