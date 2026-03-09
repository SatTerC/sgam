"""
Disturbance detection module for SGAM.

This module provides functionality to identify disturbance events (such as
harvest, fire, or pest damage) based on rapid declines in GPP and LAI.
"""

import numpy as np
from numpy.typing import NDArray


def aggregate_to_weekly(daily_severity: NDArray[np.float64]) -> NDArray[np.float64]:
    """Aggregate daily disturbance severity to weekly timesteps.

    Takes the maximum daily severity observed within each 7-day window.
    If the input length is not a multiple of 7, it is padded with zeros.

    Parameters
    ----------
    daily_severity : NDArray[np.float64]
        Array of daily disturbance severity values.

    Returns
    -------
    NDArray[np.float64]
        Array of weekly maximum disturbance severity values.
    """
    remainder = len(daily_severity) % 7
    if remainder > 0:
        daily_severity = np.pad(daily_severity, (0, 7 - remainder), mode="constant")

    weekly_max = daily_severity.reshape(-1, 7).max(axis=1)

    return weekly_max


class Disturbances:
    """Disturbance event detector.

    Identifies disturbance events based on rapid declines in GPP and LAI
    during the growing season. Disturbances include events such as
    harvest, fire, pest damage, or other biomass-removing events.

    Attributes
    ----------
    growing_season_limit : float
        Minimum temperature (°C) defining the growing season. Days with
        temperature above this threshold are considered within the growing
        season and eligible for disturbance detection.
    disturbance_threshold : float
        Fractional decline threshold (0-1). A disturbance event is detected
        when both GPP and LAI decline by more than this fraction in a single
        day.

    Example
    -------
    >>> disturbances = Disturbances(growing_season_limit=15.0, disturbance_threshold=0.3)
    >>> temperature = np.array([20.0, 20.0, 20.0, 20.0])
    >>> gpp = np.array([5.0, 5.0, 1.0, 1.0])
    >>> lai = np.array([1.0, 1.0, 0.2, 0.2])
    >>> result = disturbances.forward(temperature, gpp, lai)
    """

    def __init__(
        self,
        growing_season_limit: float,
        disturbance_threshold: float,
    ) -> None:
        """Initialize the Disturbances detector.

        Parameters
        ----------
        growing_season_limit : float
            Minimum temperature (°C) defining the growing season.
        disturbance_threshold : float
            Fractional decline threshold (0-1) for detecting disturbances.
        """
        self.growing_season_limit = growing_season_limit
        self.disturbance_threshold = disturbance_threshold

    def forward(
        self,
        temperature: NDArray[np.float64],
        gpp: NDArray[np.float64],
        lai: NDArray[np.float64],
        aggregate: bool = False,
    ) -> NDArray[np.float64]:
        """Identify disturbance events and compute their severity.

        A disturbance event is detected when all three conditions are met:
        1. Temperature exceeds the growing season threshold
        2. GPP declines by more than the disturbance threshold (fraction)
        3. LAI declines by more than the disturbance threshold (fraction)

        Note: Input data should be at DAILY timestep.

        Parameters
        ----------
        temperature : NDArray[np.float64]
            Daily temperature values (°C).
        gpp : NDArray[np.float64]
            Daily GPP values (gC m⁻² day⁻¹).
        lai : NDArray[np.float64]
            Daily LAI values (m² m⁻²).
        aggregate : bool, optional
            If True (default), aggregate daily results to weekly timesteps
            by taking the maximum within each 7-day window. If False, return
            daily values.

        Returns
        -------
        NDArray[np.float64]
            Disturbance severity as a fraction of biomass lost (0-1).
            Non-disturbance days have value 0.
            If aggregate=True, values are weekly; otherwise daily.
        """
        Δgpp = np.zeros_like(gpp)
        Δlai = np.zeros_like(lai)

        with np.errstate(divide="ignore", invalid="ignore"):
            Δgpp[1:] = (gpp[1:] - gpp[:-1]) / gpp[:-1]
            Δlai[1:] = (lai[1:] - lai[:-1]) / lai[:-1]

        disturbance_days = (
            (temperature > self.growing_season_limit)
            & (-Δgpp > self.disturbance_threshold)
            & (-Δlai > self.disturbance_threshold)
        )

        declines = np.fmax(-Δgpp, -Δlai)
        disturbance_severity = np.clip(
            np.nan_to_num(declines, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0
        )

        result_daily = disturbance_severity * disturbance_days

        return aggregate_to_weekly(result_daily) if aggregate else result_daily

    def __call__(
        self,
        temperature: NDArray[np.float64],
        gpp: NDArray[np.float64],
        lai: NDArray[np.float64],
        aggregate: bool = False,
    ) -> NDArray[np.float64]:
        """An alias for `forward`."""
        return self.forward(temperature, gpp, lai, aggregate=aggregate)
