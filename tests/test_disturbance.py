"""Tests for disturbance detection in Disturbances."""

import numpy as np
from sgam.disturbance import Disturbances


class TestDisturbanceDetection:
    """Test suite for disturbance event detection.

    Tests the Disturbances class which identifies disturbance events
    based on rapid declines in GPP and LAI.
    """

    def test_zero_disturbance_events(
        self,
        driving_data_no_disturbance,
    ):
        """Test that no disturbance events are detected when driving data is constant."""
        disturbance = Disturbances(growing_season_limit=15.0, disturbance_threshold=0.3)
        result = disturbance.forward(
            driving_data_no_disturbance["temperature"],
            driving_data_no_disturbance["gpp"],
            driving_data_no_disturbance["lai_obs"],
        )
        disturbance_count = np.count_nonzero(result)
        assert disturbance_count == 0

    def test_one_disturbance_event(
        self,
        driving_data_one_disturbance,
    ):
        """Test that exactly one disturbance event is detected."""
        disturbance = Disturbances(growing_season_limit=15.0, disturbance_threshold=0.3)
        result = disturbance.forward(
            driving_data_one_disturbance["temperature"],
            driving_data_one_disturbance["gpp"],
            driving_data_one_disturbance["lai_obs"],
        )
        disturbance_count = np.count_nonzero(result)
        assert disturbance_count == 1

    def test_two_disturbance_events(
        self,
        driving_data_two_disturbances,
    ):
        """Test that exactly two disturbance events are detected."""
        disturbance = Disturbances(growing_season_limit=15.0, disturbance_threshold=0.3)
        result = disturbance.forward(
            driving_data_two_disturbances["temperature"],
            driving_data_two_disturbances["gpp"],
            driving_data_two_disturbances["lai_obs"],
        )
        disturbance_count = np.count_nonzero(result)
        assert disturbance_count == 2
