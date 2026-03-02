"""
SGAM (Simplified Growth/GPP Allocation Model) component.

This module provides the SgamComponent class, which simulates the allocation
of gross primary productivity (GPP) to plant carbon pools (leaf, stem, root)
across different plant types (tree, grass, crop, shrub), including turnover,
respiration, and disturbance events.
"""

import numpy as np
from numpy.typing import NDArray

from .pft import PlantFunctionalType, PFT_PARAMS
from .utils import (
    compute_cue,
    compute_growing_season,
    compute_relative_changes,
    compute_drought_modifier,
    compute_allocation_percentages,
    solve_pool_recurrence,
    find_epoch_boundaries,
)


class SgamComponent:
    """
    The Simplified Growth/GPP Allocation Model (SGAM) simulates the allocation of gross primary productivity (GPP)
    to plant carbon pools (leaf, stem, root) for 4 plant types (tree, grass, crop, shrub) over time,
    based on environmental drivers and physiological parameters.
    It accounts for dynamic allocation, turnover, respiration, disturbance/harvest events, and outputs pool sizes and fluxes.

    Parameters
    ----------
        plant_type : PlantFunctionalType
            Type of plant (tree, grass, crop, or shrub).
        leaf_pool_init: float
            Initial leaf carbon pool size.
        stem_pool_init: float
            Initial stem carbon pool size.
        root_pool_init: float
            Initial root carbon pool size.
        leaf_turnover_rate : float | None, optional
            Daily turnover rate for leaf. Uses PFT default if None.
        stem_turnover_rate : float | None, optional
            Daily turnover rate for stem. Uses PFT default if None.
        root_turnover_rate : float | None, optional
            Daily turnover rate for root. Uses PFT default if None.
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
        - 'leaf', 'stem', 'root': Carbon pool sizes.
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
        leaf_pool_init: float,
        stem_pool_init: float,
        root_pool_init: float,
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
        self.leaf_pool_init = leaf_pool_init
        self.stem_pool_init = stem_pool_init
        self.root_pool_init = root_pool_init
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
            Observed leaf area index. From model_inputs.
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

        Returns
        -------
        dict[str, NDArray]
            Output dict with keys:
            - 'leaf', 'stem', 'root': Carbon pool sizes.
            - 'litter_to_soil': Daily litter carbon to soil.
            - 'leaf_respiration_loss', 'stem_respiration_loss', 'root_respiration_loss': Daily respiration losses.
            - 'leaf_area_index': Simulated LAI.
            - 'npp': Net primary productivity.
            - 'cue': Carbon use efficiency timeseries.
            - 'disturbance': Carbon loss due to disturbance/harvest.
        """
        n_timesteps = len(gpp)

        pft_params = PFT_PARAMS[self.plant_type]

        cue = compute_cue(lue, iwue)
        growing_season = compute_growing_season(temperature, self.growing_season_limit)
        gpp_rel_change = compute_relative_changes(gpp)
        lai_rel_change = compute_relative_changes(lai_obs)

        moisture_threshold = np.percentile(soil_moisture, 25)
        vpd_max = np.percentile(vpd, 75)

        (
            leaf_allocation_percentage,
            stem_allocation_percentage,
            root_allocation_percentage,
        ) = compute_allocation_percentages(
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
        allocated_gpp_stem = gpp * stem_allocation_percentage * self.timestep
        allocated_gpp_root = gpp * root_allocation_percentage * self.timestep

        leaf_respiration = allocated_gpp_leaf * (1 - cue)
        stem_respiration = allocated_gpp_stem * (1 - cue)
        root_respiration = allocated_gpp_root * (1 - cue)

        leaf_turnover_factor = 1 - (1 - self.leaf_turnover_rate) ** self.timestep
        stem_turnover_factor = 1 - (1 - self.stem_turnover_rate) ** self.timestep
        root_turnover_factor = 1 - (1 - self.root_turnover_rate) ** self.timestep

        litter_cue_modifier = 2 - cue

        disturbance_mask = (
            growing_season
            & (gpp_rel_change < -self.disturbance_limit)
            & (lai_rel_change < -self.disturbance_limit)
        )

        disturbance_fraction = np.minimum(
            np.maximum(np.abs(gpp_rel_change), np.abs(lai_rel_change)), 1.0
        )

        epochs = find_epoch_boundaries(disturbance_mask)

        leaf = np.zeros(n_timesteps)
        stem = np.zeros(n_timesteps)
        root = np.zeros(n_timesteps)
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
                current_leaf = self.leaf_pool_init
                current_stem = self.stem_pool_init
                current_root = self.root_pool_init
            else:
                current_leaf = leaf[epoch_start - 1]
                current_stem = stem[epoch_start - 1]
                current_root = root[epoch_start - 1]

            if (
                is_crop
                and gpp[epoch_start] <= 1.0
                and (current_leaf + current_stem + current_root) == 0.0
            ):
                leaf[epoch_slice] = 0.0
                stem[epoch_slice] = 0.0
                root[epoch_slice] = 0.0
                litter_to_soil[epoch_slice] = 0.0
                leaf_respiration_loss[epoch_slice] = 0.0
                stem_respiration_loss[epoch_slice] = 0.0
                root_respiration_loss[epoch_slice] = 0.0
                leaf_area_index[epoch_slice] = 0.0
                npp_out[epoch_slice] = 0.0
                continue

            litter_cue_epoch = litter_cue_modifier[epoch_slice]

            retention_factor_leaf = (
                1 - leaf_turnover_factor * litter_cue_epoch / self.timestep
            )
            retention_factor_stem = (
                1 - stem_turnover_factor * litter_cue_epoch / self.timestep
            )
            retention_factor_root = (
                1 - root_turnover_factor * litter_cue_epoch / self.timestep
            )

            net_allocation_leaf = (
                allocated_gpp_leaf[epoch_slice] - leaf_respiration[epoch_slice]
            )
            net_allocation_stem = (
                allocated_gpp_stem[epoch_slice] - stem_respiration[epoch_slice]
            )
            net_allocation_root = (
                allocated_gpp_root[epoch_slice] - root_respiration[epoch_slice]
            )

            leaf[epoch_slice] = solve_pool_recurrence(
                current_leaf, retention_factor_leaf, net_allocation_leaf
            )
            stem[epoch_slice] = solve_pool_recurrence(
                current_stem, retention_factor_stem, net_allocation_stem
            )
            root[epoch_slice] = solve_pool_recurrence(
                current_root, retention_factor_root, net_allocation_root
            )

            leaf[epoch_slice] = np.maximum(leaf[epoch_slice], 0.0)
            stem[epoch_slice] = np.maximum(stem[epoch_slice], 0.0)
            root[epoch_slice] = np.maximum(root[epoch_slice], 0.0)

            pool_before_leaf = np.empty(epoch_length)
            pool_before_stem = np.empty(epoch_length)
            pool_before_root = np.empty(epoch_length)

            pool_before_leaf[0] = current_leaf
            pool_before_stem[0] = current_stem
            pool_before_root[0] = current_root
            pool_before_leaf[1:] = leaf[epoch_slice][:-1]
            pool_before_stem[1:] = stem[epoch_slice][:-1]
            pool_before_root[1:] = root[epoch_slice][:-1]

            litter_to_soil[epoch_slice] = (
                pool_before_leaf
                * leaf_turnover_factor
                / self.timestep
                * litter_cue_epoch
                + pool_before_stem
                * stem_turnover_factor
                / self.timestep
                * litter_cue_epoch
                + pool_before_root
                * root_turnover_factor
                / self.timestep
                * litter_cue_epoch
            )

            leaf_area_index[epoch_slice] = leaf[epoch_slice] / self.leaf_carbon_area
            npp_out[epoch_slice] = (
                gpp[epoch_slice] * self.timestep
                - leaf_respiration[epoch_slice]
                - stem_respiration[epoch_slice]
                - root_respiration[epoch_slice]
            )

            leaf_respiration_loss[epoch_slice] = leaf_respiration[epoch_slice]
            stem_respiration_loss[epoch_slice] = stem_respiration[epoch_slice]
            root_respiration_loss[epoch_slice] = root_respiration[epoch_slice]

        disturbance_indices = np.where(disturbance_mask)[0]

        for i in disturbance_indices:
            if is_crop:
                disturbance[i] = leaf[i]
                total_pool = leaf[i] + stem[i] + root[i]
                litter_to_soil[i] += total_pool
                leaf[i] = 0.0
                stem[i] = 0.0
                root[i] = 0.0
            else:
                disturbance[i] = leaf[i] * disturbance_fraction[i]
                leaf[i] -= disturbance[i]

            leaf_respiration_loss[i] = 0.0
            stem_respiration_loss[i] = 0.0
            root_respiration_loss[i] = 0.0
            npp_out[i] = 0.0
            leaf_area_index[i] = leaf[i] / self.leaf_carbon_area

        return {
            "leaf": leaf,
            "stem": stem,
            "root": root,
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
        )
