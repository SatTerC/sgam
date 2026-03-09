import numpy as np
from numpy.typing import NDArray


def compute_relative_changes(x: NDArray) -> NDArray:
    """
    Compute relative changes between consecutive timesteps.

    Parameters
    ----------
    x : NDArray
        Input array.

    Returns
    -------
    NDArray
        Relative changes, with first element as 0.
    """
    Δx = np.zeros_like(x)
    Δx[1:] = (x[1:] - x[:-1]) / np.maximum(x[:-1], 1e-6)
    return Δx


class Disturbances:
    def __init__(
        self, growing_season_limit: float, disturbance_threshold: float
    ) -> None:
        self.growing_season_limit = growing_season_limit
        self.disturbance_threshold = disturbance_threshold

    def forward(
        self,
        temperature: NDArray[np.float64],
        gpp: NDArray[np.float64],
        lai: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Identify disturbance events and compute their severity.

        Note - input data is DAILY timestep.
        """
        # Compute relative day-to-day changes in GPP and observed LAI to detect disturbance events
        # These are fractional changes (e.g., -0.3 means 30% decrease from previous day)
        gpp_rel_change = compute_relative_changes(gpp)
        lai_rel_change = compute_relative_changes(lai)

        # Disturbance registered when:
        # (a) within growing season: temperature above threshold
        # (b) single-day drop in GPP exceeds threshold
        # (c) single-day drop in LAI exceeds threshold
        # Creates a boolean mask where True indicates a disturbance event
        disturbance_days = (
            (temperature > self.growing_season_limit)
            & (gpp_rel_change < -self.disturbance_threshold)
            & (lai_rel_change < -self.disturbance_threshold)
        )

        # Calculate disturbance severity as fraction of pool lost (capped at 100%)
        # Uses the maximum of GPP or LAI relative decline to estimate biomass loss
        disturbance_severity = np.minimum(
            np.maximum(np.abs(gpp_rel_change), np.abs(lai_rel_change)), 1.0
        )

        # Return array which is 0 for non-disturbance days, and severity on disturbance days
        return disturbance_severity * disturbance_days

    def __call__(self, temperature, gpp, lai):
        return self.forward(temperature, gpp, lai)
