"""Pytest fixtures for SGAM tests."""

import numpy as np
import pytest
from numpy.typing import NDArray


@pytest.fixture
def driving_data_no_disturbance() -> dict[str, NDArray[np.float64]]:
    """Driving data with zero disturbance events.

    All driving variables are held constant, so no disturbance events
    should be detected.
    """
    n = 30
    return {
        "temperature": np.array([20.0] * n),
        "vpd": np.array([500.0] * n),
        "lai_obs": np.array([1.0] * n),
        "day_of_year": np.arange(1, n + 1, dtype=float),
        "soil_moisture": np.array([0.5] * n),
        "gpp": np.array([5.0] * n),
        "iwue": np.array([100.0] * n),
        "lue": np.array([0.5] * n),
    }


@pytest.fixture
def driving_data_one_disturbance() -> dict[str, NDArray[np.float64]]:
    """Driving data with one disturbance event.

    Both GPP and LAI drop by more than 30% at timestep 15, triggering
    a single disturbance event.
    """
    n = 30
    gpp = np.array([5.0] * 15 + [0.5] * 15)
    lai_obs = np.array([1.0] * 15 + [0.1] * 15)
    return {
        "temperature": np.array([20.0] * n),
        "vpd": np.array([500.0] * n),
        "lai_obs": lai_obs,
        "day_of_year": np.arange(1, n + 1, dtype=float),
        "soil_moisture": np.array([0.5] * n),
        "gpp": gpp,
        "iwue": np.array([100.0] * n),
        "lue": np.array([0.5] * n),
    }


@pytest.fixture
def driving_data_two_disturbances() -> dict[str, NDArray[np.float64]]:
    """Driving data with two disturbance events.

    Both GPP and LAI drop by more than 30% at timesteps 10 and 20,
    triggering two disturbance events.
    """
    n = 30
    gpp = np.array([5.0] * 10 + [0.5] * 10 + [5.0] * 5 + [0.5] * 5)
    lai_obs = np.array([1.0] * 10 + [0.1] * 10 + [1.0] * 5 + [0.1] * 5)
    return {
        "temperature": np.array([20.0] * n),
        "vpd": np.array([500.0] * n),
        "lai_obs": lai_obs,
        "day_of_year": np.arange(1, n + 1, dtype=float),
        "soil_moisture": np.array([0.5] * n),
        "gpp": gpp,
        "iwue": np.array([100.0] * n),
        "lue": np.array([0.5] * n),
    }


@pytest.fixture
def initial_pools() -> dict[str, float]:
    """Initial carbon pool sizes for testing."""
    return {
        "leaf_pool_init": 1.0,
        "stem_pool_init": 1.0,
        "root_pool_init": 1.0,
    }
