"""
SGAM (Simplified Growth/GPP Allocation Model) component.

This module provides the SgamComponent class, which simulates the allocation
of gross primary productivity (GPP) to plant carbon pools (leaf_pool_size, stem_pool_size, root_pool_size)
across different plant types (tree, grass, crop, shrub), including turnover,
respiration, and disturbance events.
"""

import numpy as np
from numpy.typing import NDArray

from .pft import PlantFunctionalType, PFT_PARAMS
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

    Returns
    -------
    tuple[dict[str, NDArray], np.ndarray]
        Tuple of (output dict, dates array) with output dict containing:
        - 'leaf_pool_size', 'stem_pool_size', 'root_pool_size': Carbon pool sizes.
        - 'litter_to_soil': Daily litter carbon to soil.
        - 'leaf_respiration_loss', 'stem_respiration_loss', 'root_respiration_loss': Daily respiration losses.
        - 'leaf_area_index': Simulated LAI.
        - 'npp': Net primary productivity.
        - 'cue': Carbon use efficiency timeseries.
        - 'disturbance': Carbon loss due to disturbance/harvest.

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
        timestep: float = 1.0,
    ):
        pft_params = PFT_PARAMS[plant_type]

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
        self.timestep = timestep

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
        base_leaf: float,
        base_stem: float,
        base_root: float,
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
        base_leaf : float
            Base allocation fraction for leaf.
        base_stem : float
            Base allocation fraction for stem.
        base_root : float
            Base allocation fraction for root.

        Returns
        -------
        tuple[NDArray, NDArray, NDArray]
            Tuple with allocation percentages for 'leaf', 'stem', 'root'.
        """
        seasonality_mod = np.sin(2 * np.pi * day_of_year / 365.0)
        temp_mod = (temperature - 20) / 100

        dynamic_leaf = np.maximum(
            0, base_leaf + 0.15 * seasonality_mod + 0.1 * temp_mod
        )
        dynamic_root = np.maximum(
            0, base_root - 0.15 * seasonality_mod - 0.05 * temp_mod
        )
        dynamic_stem = np.maximum(0, base_stem - 0.05 * temp_mod)

        total_dynamic = dynamic_leaf + dynamic_stem + dynamic_root
        total_dynamic = np.maximum(total_dynamic, 1e-10)

        dynamic_leaf = dynamic_leaf / total_dynamic
        dynamic_stem = dynamic_stem / total_dynamic
        dynamic_root = dynamic_root / total_dynamic

        drought_modifier = self.compute_drought_modifier(
            soil_moisture, vpd, moisture_threshold, vpd_max
        )

        root_adjustment = drought_modifier * 0.1
        leaf_stem_adjustment = -drought_modifier * 0.1

        final_root = np.maximum(0, dynamic_root + root_adjustment)
        final_leaf = np.maximum(0, dynamic_leaf + leaf_stem_adjustment * 0.7)
        final_stem = np.maximum(0, dynamic_stem + leaf_stem_adjustment * 0.3)

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
        n_timesteps = len(gpp)

        pft_params = PFT_PARAMS[self.plant_type]

        cue = self.compute_cue(lue, iwue)
        gpp_rel_change = compute_relative_changes(gpp)
        lai_rel_change = compute_relative_changes(lai_obs)

        moisture_threshold = np.percentile(soil_moisture, 25)
        vpd_max = np.percentile(vpd, 75)

        (
            leaf_allocation_percentage,
            stem_pool_size_allocation_percentage,
            root_pool_size_allocation_percentage,
        ) = self.compute_allocation_percentages(
            temperature,
            day_of_year,
            soil_moisture,
            vpd,
            moisture_threshold,
            vpd_max,
            pft_params.leaf_base_allocation,
            pft_params.stem_base_allocation,
            pft_params.root_base_allocation,
        )

        allocated_gpp_leaf = gpp * leaf_allocation_percentage * self.timestep
        allocated_gpp_stem_pool_size = (
            gpp * stem_pool_size_allocation_percentage * self.timestep
        )
        allocated_gpp_root_pool_size = (
            gpp * root_pool_size_allocation_percentage * self.timestep
        )

        leaf_respiration = allocated_gpp_leaf * (1 - cue)
        stem_pool_size_respiration = allocated_gpp_stem_pool_size * (1 - cue)
        root_pool_size_respiration = allocated_gpp_root_pool_size * (1 - cue)

        leaf_pool_size_turnover_factor = (
            1 - (1 - self.leaf_turnover_rate) ** self.timestep
        )
        stem_pool_size_turnover_factor = (
            1 - (1 - self.stem_turnover_rate) ** self.timestep
        )
        root_pool_size_turnover_factor = (
            1 - (1 - self.root_turnover_rate) ** self.timestep
        )

        litter_cue_modifier = 2 - cue

        # Disturbance registered when:
        # (a) within growing season: temperature above threshold
        # (b) single-day drop in GPP exceeds threshold
        # (c) single-day drop in LAI exceeds threshold
        disturbance_mask = (
            (temperature > self.growing_season_limit)
            & (gpp_rel_change < -self.disturbance_limit)
            & (lai_rel_change < -self.disturbance_limit)
        )

        disturbance_fraction = np.minimum(
            np.maximum(np.abs(gpp_rel_change), np.abs(lai_rel_change)), 1.0
        )

        epochs = find_segments(disturbance_mask)

        leaf_pool_size = np.zeros(n_timesteps)
        stem_pool_size = np.zeros(n_timesteps)
        root_pool_size = np.zeros(n_timesteps)
        litter_to_soil = np.zeros(n_timesteps)
        leaf_respiration_loss = np.zeros(n_timesteps)
        stem_respiration_loss = np.zeros(n_timesteps)
        root_respiration_loss = np.zeros(n_timesteps)
        leaf_area_index = np.zeros(n_timesteps)
        npp_out = np.zeros(n_timesteps)
        disturbance = np.zeros(n_timesteps)

        is_crop = self.plant_type is PlantFunctionalType.CROP

        for epoch_start, epoch_end in epochs:
            epoch_slice = slice(epoch_start, epoch_end)
            epoch_length = epoch_end - epoch_start

            if epoch_start == 0:
                current_leaf_pool_size = leaf_pool_init
                current_stem_pool_size = stem_pool_init
                current_root_pool_size = root_pool_init
            else:
                current_leaf_pool_size = leaf_pool_size[epoch_start - 1]
                current_stem_pool_size = stem_pool_size[epoch_start - 1]
                current_root_pool_size = root_pool_size[epoch_start - 1]

            if (
                is_crop
                and gpp[epoch_start] <= 1.0
                and (
                    current_leaf_pool_size
                    + current_stem_pool_size
                    + current_root_pool_size
                )
                == 0.0
            ):
                leaf_pool_size[epoch_slice] = 0.0
                stem_pool_size[epoch_slice] = 0.0
                root_pool_size[epoch_slice] = 0.0
                litter_to_soil[epoch_slice] = 0.0
                leaf_respiration_loss[epoch_slice] = 0.0
                stem_respiration_loss[epoch_slice] = 0.0
                root_respiration_loss[epoch_slice] = 0.0
                leaf_area_index[epoch_slice] = 0.0
                npp_out[epoch_slice] = 0.0
                continue

            litter_cue_epoch = litter_cue_modifier[epoch_slice]

            retention_factor_leaf_pool_size = (
                1 - leaf_pool_size_turnover_factor * litter_cue_epoch / self.timestep
            )
            retention_factor_stem_pool_size = (
                1 - stem_pool_size_turnover_factor * litter_cue_epoch / self.timestep
            )
            retention_factor_root_pool_size = (
                1 - root_pool_size_turnover_factor * litter_cue_epoch / self.timestep
            )

            net_allocation_leaf_pool_size = (
                allocated_gpp_leaf[epoch_slice] - leaf_respiration[epoch_slice]
            )
            net_allocation_stem_pool_size = (
                allocated_gpp_stem_pool_size[epoch_slice]
                - stem_pool_size_respiration[epoch_slice]
            )
            net_allocation_root_pool_size = (
                allocated_gpp_root_pool_size[epoch_slice]
                - root_pool_size_respiration[epoch_slice]
            )

            leaf_pool_size[epoch_slice] = solve_recurrence(
                current_leaf_pool_size,
                retention_factor_leaf_pool_size,
                net_allocation_leaf_pool_size,
            )
            stem_pool_size[epoch_slice] = solve_recurrence(
                current_stem_pool_size,
                retention_factor_stem_pool_size,
                net_allocation_stem_pool_size,
            )
            root_pool_size[epoch_slice] = solve_recurrence(
                current_root_pool_size,
                retention_factor_root_pool_size,
                net_allocation_root_pool_size,
            )

            leaf_pool_size[epoch_slice] = np.maximum(leaf_pool_size[epoch_slice], 0.0)
            stem_pool_size[epoch_slice] = np.maximum(stem_pool_size[epoch_slice], 0.0)
            root_pool_size[epoch_slice] = np.maximum(root_pool_size[epoch_slice], 0.0)

            pool_before_leaf_pool_size = np.empty(epoch_length)
            pool_before_stem_pool_size = np.empty(epoch_length)
            pool_before_root_pool_size = np.empty(epoch_length)

            pool_before_leaf_pool_size[0] = current_leaf_pool_size
            pool_before_stem_pool_size[0] = current_stem_pool_size
            pool_before_root_pool_size[0] = current_root_pool_size
            pool_before_leaf_pool_size[1:] = leaf_pool_size[epoch_slice][:-1]
            pool_before_stem_pool_size[1:] = stem_pool_size[epoch_slice][:-1]
            pool_before_root_pool_size[1:] = root_pool_size[epoch_slice][:-1]

            litter_to_soil[epoch_slice] = (
                pool_before_leaf_pool_size
                * leaf_pool_size_turnover_factor
                / self.timestep
                * litter_cue_epoch
                + pool_before_stem_pool_size
                * stem_pool_size_turnover_factor
                / self.timestep
                * litter_cue_epoch
                + pool_before_root_pool_size
                * root_pool_size_turnover_factor
                / self.timestep
                * litter_cue_epoch
            )

            leaf_area_index[epoch_slice] = (
                leaf_pool_size[epoch_slice] / self.leaf_carbon_area
            )
            npp_out[epoch_slice] = (
                gpp[epoch_slice] * self.timestep
                - leaf_respiration[epoch_slice]
                - stem_pool_size_respiration[epoch_slice]
                - root_pool_size_respiration[epoch_slice]
            )

            leaf_respiration_loss[epoch_slice] = leaf_respiration[epoch_slice]
            stem_respiration_loss[epoch_slice] = stem_pool_size_respiration[epoch_slice]
            root_respiration_loss[epoch_slice] = root_pool_size_respiration[epoch_slice]

        disturbance_indices = np.where(disturbance_mask)[0]

        for i in disturbance_indices:
            if is_crop:
                disturbance[i] = leaf_pool_size[i]
                total_pool = leaf_pool_size[i] + stem_pool_size[i] + root_pool_size[i]
                litter_to_soil[i] += total_pool
                leaf_pool_size[i] = 0.0
                stem_pool_size[i] = 0.0
                root_pool_size[i] = 0.0
            else:
                disturbance[i] = leaf_pool_size[i] * disturbance_fraction[i]
                leaf_pool_size[i] -= disturbance[i]

            leaf_respiration_loss[i] = 0.0
            stem_respiration_loss[i] = 0.0
            root_respiration_loss[i] = 0.0
            npp_out[i] = 0.0
            leaf_area_index[i] = leaf_pool_size[i] / self.leaf_carbon_area

        return {
            "leaf_pool_size": leaf_pool_size,
            "stem_pool_size": stem_pool_size,
            "root_pool_size": root_pool_size,
            "litter_to_soil": litter_to_soil,
            "leaf_respiration_loss": leaf_respiration_loss,
            "stem_respiration_loss": stem_respiration_loss,
            "root_respiration_loss": root_respiration_loss,
            "leaf_area_index": leaf_area_index,
            "npp": npp_out,
            "cue": cue,
            "disturbance": disturbance,
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
