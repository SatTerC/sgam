"""
SGAM (Simplified Growth/GPP Allocation Model) component.

This module provides the SgamComponent class, which simulates the allocation
of gross primary productivity (GPP) to plant carbon pools (leaf_pool_size, stem_pool_size, root_pool_size)
across different plant types (tree, grass, crop, shrub), including turnover,
respiration, and disturbance events.
"""

import numpy as np
from numpy.typing import NDArray

from .pft import PlantFunctionalType, get_default_pft_params
from .utils import (
    rescale_to_unit_interval,
    compute_relative_changes,
    solve_recurrence,
    find_segments,
)


class SgamComponent:
    """
    The Simplified Growth/GPP Allocation Model (SGAM) simulates the allocation of gross primary productivity (GPP)
    to plant carbon pools (leaf_pool_size, stem_pool_size, root_pool_size) for 4 plant types (tree, grass, crop, shrub) over time,
    based on environmental drivers and physiological parameters.
    It accounts for dynamic allocation, turnover, respiration, disturbance/harvest events, and outputs pool sizes and fluxes.

    Parameters
    ----------
        plant_type : PlantFunctionalType
            Type of plant (tree, grass, crop, or shrub).
        leaf_turnover_rate : float | None, optional
            Daily turnover rate for leaf_pool_size. Uses PFT default if None.
        stem_turnover_rate : float | None, optional
            Daily turnover rate for stem_pool_size. Uses PFT default if None.
        root_turnover_rate : float | None, optional
            Daily turnover rate for root_pool_size. Uses PFT default if None.
        leaf_carbon_area : float | None, optional
            Leaf carbon area conversion factor. Uses PFT default if None.
        disturbance_limit : float | None, optional
            Threshold for detecting disturbance/harvest events. Uses PFT default if None.
        growing_season_limit : float | None, optional
            Minimum temperature (degC) for growing season. Uses PFT default if None.
        timestep : float, optional
            Timestep in days (default: 1.0).

    Notes
    -----
    - Pools and fluxes are updated daily based on allocation rules, turnover, and disturbance detection.
    - For crops, pools are reset to zero at harvest events.

    Todo
    -----
    - Refine crop modelling --> growing_season_limit necessary ?
    - Add PC output for RothC when crop harvested or not emerged
    - Add grazing -> manure return to RothC
    """

    def __init__(
        self,
        plant_type: PlantFunctionalType,
        *,
        leaf_turnover_rate: float | None = None,
        stem_turnover_rate: float | None = None,
        root_turnover_rate: float | None = None,
        leaf_carbon_area: float | None = None,
        disturbance_limit: float | None = None,
        growing_season_limit: float | None = None,
        leaf_base_allocation: float | None = None,
        stem_base_allocation: float | None = None,
        root_base_allocation: float | None = None,
        timestep: float = 1.0,
    ):
        pft_params = get_default_pft_params(plant_type)

        self.plant_type = plant_type
        self.leaf_turnover_rate = (
            leaf_turnover_rate
            if leaf_turnover_rate is not None
            else pft_params.leaf_turnover_rate
        )
        self.stem_turnover_rate = (
            stem_turnover_rate
            if stem_turnover_rate is not None
            else pft_params.stem_turnover_rate
        )
        self.root_turnover_rate = (
            root_turnover_rate
            if root_turnover_rate is not None
            else pft_params.root_turnover_rate
        )
        self.leaf_carbon_area = (
            leaf_carbon_area
            if leaf_carbon_area is not None
            else pft_params.leaf_carbon_area
        )
        self.disturbance_limit = (
            disturbance_limit
            if disturbance_limit is not None
            else pft_params.disturbance_limit
        )
        self.growing_season_limit = (
            growing_season_limit
            if growing_season_limit is not None
            else pft_params.growing_season_limit
        )
        self.leaf_base_allocation = (
            leaf_base_allocation
            if leaf_base_allocation is not None
            else pft_params.leaf_base_allocation
        )
        self.stem_base_allocation = (
            stem_base_allocation
            if stem_base_allocation is not None
            else pft_params.stem_base_allocation
        )
        self.root_base_allocation = (
            root_base_allocation
            if root_base_allocation is not None
            else pft_params.root_base_allocation
        )
        self.timestep = timestep

        # Calculate turnover factors for each tissue pool using the exponential decay formula
        # Turnover factor represents the fraction of tissue that dies/turns over per timestep
        # Formula: 1 - (1 - rate)^timestep converts daily rate to effective per-timestep turnover
        self.leaf_turnover_factor = 1 - (1 - self.leaf_turnover_rate) ** self.timestep
        self.stem_turnover_factor = 1 - (1 - self.stem_turnover_rate) ** self.timestep
        self.root_turnover_factor = 1 - (1 - self.root_turnover_rate) ** self.timestep

    def compute_cue(self, lue: NDArray, iwue: NDArray) -> NDArray:
        """
        Compute carbon use efficiency (CUE) from light use efficiency and
        intrinsic water use efficiency.

        Parameters
        ----------
        lue : NDArray
            Light use efficiency values.
        iwue : NDArray
            Intrinsic water use efficiency values.

        Returns
        -------
        NDArray
            Carbon use efficiency values.
        """
        lue_norm = rescale_to_unit_interval(lue)
        iwue_norm = rescale_to_unit_interval(iwue)
        iwue_norm_inv = 1 - iwue_norm
        cue_raw = 0.5 * (lue_norm + iwue_norm_inv)
        return 0.2 + cue_raw * (0.9 - 0.2)

    def compute_drought_modifier(
        self,
        soil_moisture: NDArray,
        vpd: NDArray,
        moisture_threshold: float,
        vpd_max: float,
    ) -> NDArray:
        """
        Compute drought modifier based on soil moisture and VPD.

        Parameters
        ----------
        soil_moisture : NDArray
            Soil moisture values.
        vpd : NDArray
            Vapor pressure deficit values (Pa).
        moisture_threshold : float
            Threshold for soil moisture stress.
        vpd_max : float
            Maximum VPD value.

        Returns
        -------
        NDArray
            Drought modifier values.
        """
        normalized_moisture = np.minimum(soil_moisture / moisture_threshold, 1.0)
        normalized_vpd = np.minimum(vpd / vpd_max, 1.0)
        return (1 - normalized_moisture) + normalized_vpd

    def compute_allocation_percentages(
        self,
        temperature: NDArray,
        day_of_year: NDArray,
        soil_moisture: NDArray,
        vpd: NDArray,
        moisture_threshold: float,
        vpd_max: float,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """
        Compute dynamic allocation percentages based on environmental factors.

        Parameters
        ----------
        temperature : NDArray
            Temperature values (degC).
        day_of_year : NDArray
            Day of year values.
        soil_moisture : NDArray
            Soil moisture values.
        vpd : NDArray
            Vapor pressure deficit values (Pa).
        moisture_threshold : float
            Threshold for soil moisture stress.
        vpd_max : float
            Maximum VPD value.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            Tuple with allocation percentages for 'leaf', 'stem', 'root'.
        """
        # NOTE: a lot of hard-coded values here - needs looking at

        # Calculate seasonal and temperature modifiers for allocation fractions
        # seasonality_mod: sinusoidal variation with annual cycle (peak at summer solstice)
        # temp_mod: normalized temperature deviation from 20°C (warmer = more to leaves)
        seasonality_mod = np.sin(2 * np.pi * day_of_year / 365.0)
        temp_mod = (temperature - 20) / 100

        # Apply modifiers to base allocation fractions to get dynamic allocations
        # Positive seasonality: more to leaves in summer, more to roots in winter
        # Positive temp_mod: warmer temps favor leaf allocation over stem
        dynamic_leaf = np.maximum(
            0, self.leaf_base_allocation + 0.15 * seasonality_mod + 0.1 * temp_mod
        )
        dynamic_root = np.maximum(
            0, self.root_base_allocation - 0.15 * seasonality_mod - 0.05 * temp_mod
        )
        dynamic_stem = np.maximum(0, self.stem_base_allocation - 0.05 * temp_mod)

        # Normalize dynamic allocations so they sum to 1.0 (preserving relative proportions)
        total_dynamic = dynamic_leaf + dynamic_stem + dynamic_root
        total_dynamic = np.maximum(total_dynamic, 1e-10)
        dynamic_leaf = dynamic_leaf / total_dynamic
        dynamic_stem = dynamic_stem / total_dynamic
        dynamic_root = dynamic_root / total_dynamic

        # Compute drought stress modifier based on soil moisture and VPD
        # Values > 0 indicate drought stress (0 = no stress, higher = more stress)
        drought_modifier = self.compute_drought_modifier(
            soil_moisture, vpd, moisture_threshold, vpd_max
        )

        # Under drought conditions, shift allocation toward roots and away from leaves/stems
        # Root adjustment: +10% allocation under full drought
        # Leaf+stem adjustment: -10% total under full drought (70% to leaves, 30% to stems)
        root_adjustment = drought_modifier * 0.1
        leaf_stem_adjustment = -drought_modifier * 0.1

        # Apply drought adjustments and ensure non-negative allocations
        final_root = np.maximum(0, dynamic_root + root_adjustment)
        final_leaf = np.maximum(0, dynamic_leaf + leaf_stem_adjustment * 0.7)
        final_stem = np.maximum(0, dynamic_stem + leaf_stem_adjustment * 0.3)

        # Final normalization to ensure allocations sum to 1.0
        total_percentage = final_leaf + final_stem + final_root
        total_percentage = np.maximum(total_percentage, 1e-10)

        return (
            final_leaf / total_percentage,
            final_stem / total_percentage,
            final_root / total_percentage,
        )

    def forward(
        self,
        temperature: NDArray[np.float64],
        vpd: NDArray[np.float64],
        lai_obs: NDArray[np.float64],
        day_of_year: NDArray[np.float64],
        soil_moisture: NDArray[np.float64],
        gpp: NDArray[np.float64],
        iwue: NDArray[np.float64],
        lue: NDArray[np.float64],
        leaf_pool_init: float,
        stem_pool_init: float,
        root_pool_init: float,
    ) -> dict[str, NDArray]:
        """
        Compute growth and allocation.

        Parameters
        ----------
        temperature : NDArray[np.float64]
            Air temperature (degrees Celsius). From model_inputs.
        vpd : NDArray[np.float64]
            Vapor pressure deficit (Pascals). From model_inputs.
        lai_obs : NDArray[np.float64]
            Observed leaf_pool_size area index. From model_inputs.
        day_of_year : NDArray[np.float64]
            Day of year. From model_inputs.
        soil_moisture : NDArray[np.float64]
            Soil moisture content (mm). From water_outputs.
        gpp : NDArray[np.float64]
            Gross primary productivity. From productivity_outputs.
        iwue : NDArray[np.float64]
            Intrinsic water use efficiency. From productivity_outputs.
        lue : NDArray[np.float64]
            Light use efficiency. From productivity_outputs.
        leaf_pool_init : float
            Initial leaf carbon pool size.
        stem_pool_init : float
            Initial stem carbon pool size.
        root_pool_init : float
            Initial root carbon pool size.

        Returns
        -------
        dict[str, NDArray]
            Output dict with keys:
            - 'leaf_pool_size', 'stem_pool_size', 'root_pool_size': Carbon pool sizes.
            - 'litter_to_soil': Daily litter carbon to soil.
            - 'leaf_respiration_loss', 'stem_respiration_loss', 'root_respiration_loss': Daily respiration losses.
            - 'leaf_area_index': Simulated LAI.
            - 'npp': Net primary productivity.
            - 'cue': Carbon use efficiency timeseries.
            - 'disturbance': Carbon loss due to disturbance/harvest.
        """
        # Determine number of simulation timesteps from GPP input array
        n_timesteps = len(gpp)

        # Compute Carbon Use Efficiency (CUE) from light use efficiency (LUE) and intrinsic water use efficiency (IWUE)
        # CUE represents the fraction of GPP that remains as Net Primary Productivity (NPP) after autotrophic respiration
        cue = self.compute_cue(lue, iwue)

        # Compute relative day-to-day changes in GPP and observed LAI to detect disturbance events
        # These are fractional changes (e.g., -0.3 means 30% decrease from previous day)
        gpp_rel_change = compute_relative_changes(gpp)
        lai_rel_change = compute_relative_changes(lai_obs)

        # Calculate environmental stress thresholds from percentiles of input data
        # 25th percentile of soil moisture serves as moisture stress threshold
        moisture_threshold = np.percentile(soil_moisture, 25)
        # 75th percentile of VPD serves as maximum vapor pressure deficit threshold
        vpd_max = np.percentile(vpd, 75)
        # NOTE: should these be configurable by the user??

        # Compute dynamic carbon allocation fractions for each tissue type (leaf, stem, root)
        # These percentages vary seasonally and in response to environmental conditions (temperature, moisture, VPD)
        (
            leaf_allocation_percentage,
            stem_allocation_percentage,
            root_allocation_percentage,
        ) = self.compute_allocation_percentages(
            temperature,
            day_of_year,
            soil_moisture,
            vpd,
            moisture_threshold,
            vpd_max,
        )

        # Allocate GPP to each tissue pool based on computed percentages and timestep
        # This represents the portion of gross primary productivity assigned to each carbon pool
        leaf_allocated_gpp = gpp * leaf_allocation_percentage * self.timestep
        stem_allocated_gpp = gpp * stem_allocation_percentage * self.timestep
        root_allocated_gpp = gpp * root_allocation_percentage * self.timestep

        # Calculate autotrophic respiration losses for each pool using CUE
        # (1 - CUE) represents the respiration fraction; respiration uses oxygen and releases CO2
        leaf_respiration_loss = leaf_allocated_gpp * (1 - cue)
        stem_respiration_loss = stem_allocated_gpp * (1 - cue)
        root_respiration_loss = root_allocated_gpp * (1 - cue)

        # Calculate net carbon allocation to each pool (GPP allocation minus respiration losses)
        # This is the amount of carbon that actually gets added to each pool
        leaf_net_allocation = leaf_allocated_gpp - leaf_respiration_loss
        stem_net_allocation = stem_allocated_gpp - stem_respiration_loss
        root_net_allocation = root_allocated_gpp - root_respiration_loss

        # NPP = GPP - total autotrophic respiration
        npp_out = (
            gpp * self.timestep
            - leaf_respiration_loss
            - stem_respiration_loss
            - root_respiration_loss
        )
        # BUG: Possible bug - I'm not at all convinced by the timestep multiplier here.
        # What are the dimensions of respiration loss, vs gpp?

        # Compute litter CUE modifier: higher when CUE is low (more carbon lost to respiration becomes litter)
        # When CUE is high (0.9), modifier is 1.1; when CUE is low (0.2), modifier is 1.8
        litter_cue_modifier = 2 - cue

        # Pre-calculate decay factors for each pool
        leaf_decay_factor = (
            self.leaf_turnover_factor / self.timestep
        ) * litter_cue_modifier
        stem_decay_factor = (
            self.stem_turnover_factor / self.timestep
        ) * litter_cue_modifier
        root_decay_factor = (
            self.root_turnover_factor / self.timestep
        ) * litter_cue_modifier

        # Calculate retention factors for each tissue pool
        # Retention factor = 1 - (turnover_fraction * litter_cue_modifier / timestep)
        # Higher turnover or lower CUE means less carbon is retained in the pool
        leaf_retention_factor = 1 - leaf_decay_factor
        stem_retention_factor = 1 - stem_decay_factor
        root_retention_factor = 1 - root_decay_factor

        # Disturbance registered when:
        # (a) within growing season: temperature above threshold
        # (b) single-day drop in GPP exceeds threshold
        # (c) single-day drop in LAI exceeds threshold
        # Creates a boolean mask where True indicates a disturbance event
        disturbance_mask = (
            (temperature > self.growing_season_limit)
            & (gpp_rel_change < -self.disturbance_limit)
            & (lai_rel_change < -self.disturbance_limit)
        )

        # Calculate disturbance severity as fraction of pool lost (capped at 100%)
        # Uses the maximum of GPP or LAI relative decline to estimate biomass loss
        disturbance_fraction = np.minimum(
            np.maximum(np.abs(gpp_rel_change), np.abs(lai_rel_change)), 1.0
        )

        # Identify continuous epochs (time segments) between disturbance events
        # Each epoch is a period where the system operates without disturbance
        # Returns list of (start_index, end_index) tuples for each epoch
        epochs = find_segments(disturbance_mask)

        # Initialize output arrays for all state variables and fluxes
        # All arrays sized to number of timesteps, initialized to zeros
        leaf_pool_size = np.zeros(n_timesteps)
        stem_pool_size = np.zeros(n_timesteps)
        root_pool_size = np.zeros(n_timesteps)
        disturbance_losses = np.zeros(n_timesteps)

        # Initialize epoch boundary pool values to initial conditions
        # These are updated at the end of each epoch to carry state to the next epoch
        leaf_pool_epoch_init = leaf_pool_init
        stem_pool_epoch_init = stem_pool_init
        root_pool_epoch_init = root_pool_init

        # Process each epoch sequentially - an epoch is a period between disturbance events
        for epoch_start, epoch_end in epochs:
            epoch_slice = slice(epoch_start, epoch_end)

            # Solve the recurrence relation for each pool over the epoch timestep
            # Uses analytical solution: P(t) = P(0) * retention^t + sum(net_allocation * retention^(t-i))
            # This models the exponential decay of pool size with ongoing carbon inputs
            leaf_pool_size[epoch_slice] = solve_recurrence(
                leaf_pool_epoch_init,
                leaf_retention_factor[epoch_slice],
                leaf_net_allocation[epoch_slice],
            )
            stem_pool_size[epoch_slice] = solve_recurrence(
                stem_pool_epoch_init,
                stem_retention_factor[epoch_slice],
                stem_net_allocation[epoch_slice],
            )
            root_pool_size[epoch_slice] = solve_recurrence(
                root_pool_epoch_init,
                root_retention_factor[epoch_slice],
                root_net_allocation[epoch_slice],
            )

            # Ensure pools cannot go negative
            leaf_pool_size[epoch_slice] = np.maximum(leaf_pool_size[epoch_slice], 0.0)
            stem_pool_size[epoch_slice] = np.maximum(stem_pool_size[epoch_slice], 0.0)
            root_pool_size[epoch_slice] = np.maximum(root_pool_size[epoch_slice], 0.0)

            # Apply disturbance at the last timestep of the epoch, but only if it's actually a disturbance day
            if disturbance_mask[epoch_end - 1]:
                disturbance_index = epoch_end - 1

                if self.plant_type is PlantFunctionalType.CROP:
                    # Total biomass (including stem and root) becomes litter to soil
                    disturbance_losses[disturbance_index] = (
                        leaf_pool_size[disturbance_index]
                        + stem_pool_size[disturbance_index]
                        + root_pool_size[disturbance_index]
                    )
                    # For crops: harvest event removes all leaf, stem and root biomass
                    leaf_pool_size[disturbance_index] = 0.0
                    stem_pool_size[disturbance_index] = 0.0
                    root_pool_size[disturbance_index] = 0.0

                else:
                    # For non-crops: disturbance (fire, wind, etc.) removes a fraction of leaf biomass
                    # Fraction based on severity of GPP/LAI drop
                    disturbance_losses[disturbance_index] = (
                        leaf_pool_size[disturbance_index]
                        * disturbance_fraction[disturbance_index]
                    )
                    leaf_pool_size[disturbance_index] -= disturbance_losses[
                        disturbance_index
                    ]

                # After disturbance: no respiration, no NPP, LAI recalculated from remaining leaf pool
                leaf_respiration_loss[disturbance_index] = 0.0
                stem_respiration_loss[disturbance_index] = 0.0
                root_respiration_loss[disturbance_index] = 0.0
                npp_out[disturbance_index] = 0.0

            # Update epoch boundary values for next epoch
            leaf_pool_epoch_init = leaf_pool_size[epoch_end - 1]
            stem_pool_epoch_init = stem_pool_size[epoch_end - 1]
            root_pool_epoch_init = root_pool_size[epoch_end - 1]

        # Shift pools by one timestep to get "starting biomass" for each day
        leaf_pool_size_shifted = np.insert(leaf_pool_size[:-1], 0, leaf_pool_init)
        stem_pool_size_shifted = np.insert(stem_pool_size[:-1], 0, stem_pool_init)
        root_pool_size_shifted = np.insert(root_pool_size[:-1], 0, root_pool_init)

        # Calculate natural turnover
        # Litter = pool_size_before * turnover_factor * CUE_modifier
        # Represents carbon from senesced tissue that enters the soil as organic matter
        litter_to_soil = (
            leaf_pool_size_shifted * leaf_decay_factor
            + stem_pool_size_shifted * stem_decay_factor
            + root_pool_size_shifted * root_decay_factor
        )

        # Calculate prognostic leaf area index (LAI) from leaf carbon pool
        # LAI = leaf_carbon / leaf_carbon_area_conversion_factor
        # Represents the one-sided leaf area per unit ground area (m²/m²)
        lai_out = leaf_pool_size / self.leaf_carbon_area

        # Return all computed state variables and fluxes as a dictionary
        return {
            "leaf_pool_size": leaf_pool_size,
            "stem_pool_size": stem_pool_size,
            "root_pool_size": root_pool_size,
            "leaf_respiration_loss": leaf_respiration_loss,
            "stem_respiration_loss": stem_respiration_loss,
            "root_respiration_loss": root_respiration_loss,
            "litter_to_soil": litter_to_soil,
            "disturbance_losses": disturbance_losses,
            "leaf_area_index": lai_out,
            "npp": npp_out,
            "cue": cue,
        }

    def __call__(
        self,
        temperature: NDArray[np.float64],
        vpd: NDArray[np.float64],
        lai_obs: NDArray[np.float64],
        day_of_year: NDArray[np.float64],
        soil_moisture: NDArray[np.float64],
        gpp: NDArray[np.float64],
        iwue: NDArray[np.float64],
        lue: NDArray[np.float64],
        leaf_pool_init: float,
        stem_pool_init: float,
        root_pool_init: float,
    ) -> dict[str, NDArray]:
        """Alias for `forward`."""
        return self.forward(
            soil_moisture=soil_moisture,
            gpp=gpp,
            iwue=iwue,
            lue=lue,
            temperature=temperature,
            vpd=vpd,
            lai_obs=lai_obs,
            day_of_year=day_of_year,
            leaf_pool_init=leaf_pool_init,
            stem_pool_init=stem_pool_init,
            root_pool_init=root_pool_init,
        )
