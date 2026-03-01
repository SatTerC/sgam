"""
SGAM (Simplified Growth/GPP Allocation Model) component.

This module provides the SgamComponent class, which simulates the allocation
of gross primary productivity (GPP) to plant carbon pools (leaves, stem, roots)
across different plant types (tree, grass, crop, shrub), including turnover,
respiration, and disturbance events.
"""

import numpy as np
from numpy.typing import NDArray

from .utils import (
    compute_cue,
    compute_growing_season,
    compute_relative_changes,
    compute_drought_modifier,
    get_allocation_bases,
    compute_allocation_percentages,
    solve_pool_recurrence,
    find_epoch_boundaries,
)


class SgamComponent:
    """
    The Simplified Growth/GPP Allocation Model (SGAM) simulates the allocation of gross primary productivity (GPP)
    to plant carbon pools (leaves, stem, roots) for 4 plant types (tree, grass, crop, shrub) over time,
    based on environmental drivers and physiological parameters.
    It accounts for dynamic allocation, turnover, respiration, disturbance/harvest events, and outputs pool sizes and fluxes.

    Parameters
    ----------
        plant_type : str
            Type of plant ('tree', 'grass', 'crop', or 'shrub').
        leaf_pool_init: float
            Initial leaf carbon pool size.
        stem_pool_init: float
            Initial stem carbon pool size.
        root_pool_init: float
            Initial root carbon pool size.
        leaf_turnover_rate : float, optional
            Daily turnover rate for leaves (default: 0.01).
        stem_turnover_rate : float, optional
            Daily turnover rate for stem (default: 0.0001).
        root_turnover_rate : float, optional
            Daily turnover rate for roots (default: 0.005).
        leaf_carbon_area : float, optional
            Leaf carbon area conversion factor (default: 30.0).
        disturbance_limit : float, optional
            Threshold for detecting disturbance/harvest events (default: 0.3).
        growing_season_limit : float, optional
            Minimum temperature (degC) for growing season (default: 10).

    Returns
    -------
    tuple[dict[str, NDArray], np.ndarray]
        Tuple of (output dict, dates array) with output dict containing:
        - 'leaves', 'stem', 'roots': Carbon pool sizes.
        - 'litter2soil': Daily litter carbon to soil.
        - 'leaves_respiration_loss', 'stem_respiration_loss', 'roots_respiration_loss': Daily respiration losses.
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
        plant_type: str,
        leaf_pool_init: float,
        stem_pool_init: float,
        root_pool_init: float,
        leaf_turnover_rate: float = 0.01,
        stem_turnover_rate: float = 0.0001,
        root_turnover_rate: float = 0.005,
        leaf_carbon_area: float = 30.0,
        disturbance_limit: float = 0.3,
        growing_season_limit: float = 10.0,
        timestep: float = 1.0,
    ):
        self.plant_type = plant_type
        self.leaf_pool_init = leaf_pool_init
        self.stem_pool_init = stem_pool_init
        self.root_pool_init = root_pool_init
        self.leaf_turnover_rate = leaf_turnover_rate
        self.stem_turnover_rate = stem_turnover_rate
        self.root_turnover_rate = root_turnover_rate
        self.leaf_carbon_area = leaf_carbon_area
        self.disturbance_limit = disturbance_limit
        self.growing_season_limit = growing_season_limit
        self.timestep = timestep

    def __call__(
        self,
        temp_degC: NDArray[np.float64],
        vpd_Pa: NDArray[np.float64],
        lai_obs: NDArray[np.float64],
        dayofyear: NDArray[np.float64],
        soil_moisture: NDArray[np.float64],
        gpp: NDArray[np.float64],
        iwue: NDArray[np.float64],
        lue: NDArray[np.float64],
    ) -> dict[str, NDArray]:
        """
        Run the SGAM component.

        Parameters
        ----------
        temp_degC : NDArray[np.float64]
            Air temperature (degrees Celsius). From model_inputs.
        vpd_Pa : NDArray[np.float64]
            Vapor pressure deficit (Pascals). From model_inputs.
        lai_obs : NDArray[np.float64]
            Observed leaf area index. From model_inputs.
        dayofyear : NDArray[np.float64]
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
            - 'leaves', 'stem', 'roots': Carbon pool sizes.
            - 'litter2soil': Daily litter carbon to soil.
            - 'leaves_respiration_loss', 'stem_respiration_loss',_loss': Daily respiration 'roots_respiration losses.
            - 'leaf_area_index': Simulated LAI.
            - 'npp': Net primary productivity.
            - 'cue': Carbon use efficiency timeseries.
            - 'disturbance': Carbon loss due to disturbance/harvest.
        """
        outputs = self._run_sgam(
            soil_moisture=soil_moisture,
            gpp=gpp,
            iwue=iwue,
            lue=lue,
            temp=temp_degC,
            vpd=vpd_Pa,
            lai_obs=lai_obs,
            doy=dayofyear,
            ts=self.timestep,
        )

        return outputs

    def _run_sgam(
        self,
        soil_moisture: NDArray,
        gpp: NDArray,
        iwue: NDArray,
        lue: NDArray,
        temp: NDArray,
        vpd: NDArray,
        lai_obs: NDArray,
        doy: NDArray,
        ts: float = 1.0,
    ) -> dict[str, NDArray]:
        """
        Run the SGAM model for a single year of data.

        Parameters
        ----------
        soil_moisture : NDArray
            Soil moisture values.
        gpp : NDArray
            Gross primary productivity values.
        iwue : NDArray
            Intrinsic water use efficiency values.
        lue : NDArray
            Light use efficiency values.
        temp : NDArray
            Temperature values (degC).
        vpd : NDArray
            Vapor pressure deficit values (Pa).
        lai_obs : NDArray
            Observed leaf area index values.
        doy : NDArray
            Day of year values.
        ts : float
            Timestep in days.

        Returns
        -------
        dict[str, NDArray]
            Dictionary containing pool sizes and fluxes.
        """
        n = len(gpp)

        plant_type_lower = self.plant_type.lower()
        base_leaves, base_stem, base_roots = get_allocation_bases(plant_type_lower)

        cue = compute_cue(lue, iwue)
        growing_season = compute_growing_season(temp, self.growing_season_limit)
        gpp_rel_change = compute_relative_changes(gpp)
        lai_rel_change = compute_relative_changes(lai_obs)

        moisture_threshold = np.percentile(soil_moisture, 25)
        vpd_max = np.percentile(vpd, 75)

        allocation = compute_allocation_percentages(
            temp,
            doy,
            soil_moisture,
            vpd,
            moisture_threshold,
            vpd_max,
            base_leaves,
            base_stem,
            base_roots,
        )

        leaves_alloc_pct = allocation["leaves"]
        stem_alloc_pct = allocation["stem"]
        roots_alloc_pct = allocation["roots"]

        allocated_gpp_leaves = gpp * leaves_alloc_pct * ts
        allocated_gpp_stem = gpp * stem_alloc_pct * ts
        allocated_gpp_roots = gpp * roots_alloc_pct * ts

        leaves_resp = allocated_gpp_leaves * (1 - cue)
        stem_resp = allocated_gpp_stem * (1 - cue)
        roots_resp = allocated_gpp_roots * (1 - cue)

        turnover_factor_leaves = 1 - (1 - self.leaf_turnover_rate) ** ts
        turnover_factor_stem = 1 - (1 - self.stem_turnover_rate) ** ts
        turnover_factor_roots = 1 - (1 - self.root_turnover_rate) ** ts

        litter_cue_modifier = 2 - cue

        disturbance_mask = (
            growing_season
            & (gpp_rel_change < -self.disturbance_limit)
            & (lai_rel_change < -self.disturbance_limit)
        )

        frac = np.minimum(
            np.maximum(np.abs(gpp_rel_change), np.abs(lai_rel_change)), 1.0
        )

        epochs = find_epoch_boundaries(disturbance_mask)

        leaves = np.zeros(n)
        stem = np.zeros(n)
        roots = np.zeros(n)
        litter2soil = np.zeros(n)
        leaves_resp_loss = np.zeros(n)
        stem_resp_loss = np.zeros(n)
        roots_resp_loss = np.zeros(n)
        leaf_area_index = np.zeros(n)
        npp_out = np.zeros(n)
        disturbance = np.zeros(n)

        is_crop = plant_type_lower == "crop"

        for epoch_start, epoch_end in epochs:
            epoch_slice = slice(epoch_start, epoch_end)
            epoch_length = epoch_end - epoch_start

            if epoch_start == 0:
                current_leaves = self.leaf_pool_init
                current_stem = self.stem_pool_init
                current_roots = self.root_pool_init
            else:
                current_leaves = leaves[epoch_start - 1]
                current_stem = stem[epoch_start - 1]
                current_roots = roots[epoch_start - 1]

            if (
                is_crop
                and gpp[epoch_start] <= 1.0
                and (current_leaves + current_stem + current_roots) == 0.0
            ):
                leaves[epoch_slice] = 0.0
                stem[epoch_slice] = 0.0
                roots[epoch_slice] = 0.0
                litter2soil[epoch_slice] = 0.0
                leaves_resp_loss[epoch_slice] = 0.0
                stem_resp_loss[epoch_slice] = 0.0
                roots_resp_loss[epoch_slice] = 0.0
                leaf_area_index[epoch_slice] = 0.0
                npp_out[epoch_slice] = 0.0
                continue

            lcm_epoch = litter_cue_modifier[epoch_slice]

            a_leaves = 1 - turnover_factor_leaves * lcm_epoch / ts
            a_stem = 1 - turnover_factor_stem * lcm_epoch / ts
            a_roots = 1 - turnover_factor_roots * lcm_epoch / ts

            b_leaves = allocated_gpp_leaves[epoch_slice] - leaves_resp[epoch_slice]
            b_stem = allocated_gpp_stem[epoch_slice] - stem_resp[epoch_slice]
            b_roots = allocated_gpp_roots[epoch_slice] - roots_resp[epoch_slice]

            leaves[epoch_slice] = solve_pool_recurrence(
                current_leaves, a_leaves, b_leaves
            )
            stem[epoch_slice] = solve_pool_recurrence(current_stem, a_stem, b_stem)
            roots[epoch_slice] = solve_pool_recurrence(current_roots, a_roots, b_roots)

            leaves[epoch_slice] = np.maximum(leaves[epoch_slice], 0.0)
            stem[epoch_slice] = np.maximum(stem[epoch_slice], 0.0)
            roots[epoch_slice] = np.maximum(roots[epoch_slice], 0.0)

            pool_before_leaves = np.empty(epoch_length)
            pool_before_stem = np.empty(epoch_length)
            pool_before_roots = np.empty(epoch_length)

            pool_before_leaves[0] = current_leaves
            pool_before_stem[0] = current_stem
            pool_before_roots[0] = current_roots
            pool_before_leaves[1:] = leaves[epoch_slice][:-1]
            pool_before_stem[1:] = stem[epoch_slice][:-1]
            pool_before_roots[1:] = roots[epoch_slice][:-1]

            litter2soil[epoch_slice] = (
                pool_before_leaves * turnover_factor_leaves / ts * lcm_epoch
                + pool_before_stem * turnover_factor_stem / ts * lcm_epoch
                + pool_before_roots * turnover_factor_roots / ts * lcm_epoch
            )

            leaf_area_index[epoch_slice] = leaves[epoch_slice] / self.leaf_carbon_area
            npp_out[epoch_slice] = (
                gpp[epoch_slice] * ts
                - leaves_resp[epoch_slice]
                - stem_resp[epoch_slice]
                - roots_resp[epoch_slice]
            )

            leaves_resp_loss[epoch_slice] = leaves_resp[epoch_slice]
            stem_resp_loss[epoch_slice] = stem_resp[epoch_slice]
            roots_resp_loss[epoch_slice] = roots_resp[epoch_slice]

        disturbance_indices = np.where(disturbance_mask)[0]

        for i in disturbance_indices:
            if is_crop:
                disturbance[i] = leaves[i]
                total_pool = leaves[i] + stem[i] + roots[i]
                litter2soil[i] += total_pool
                leaves[i] = 0.0
                stem[i] = 0.0
                roots[i] = 0.0
            else:
                disturbance[i] = leaves[i] * frac[i]
                leaves[i] -= disturbance[i]

            leaves_resp_loss[i] = 0.0
            stem_resp_loss[i] = 0.0
            roots_resp_loss[i] = 0.0
            npp_out[i] = 0.0
            leaf_area_index[i] = leaves[i] / self.leaf_carbon_area

        return {
            "leaves": leaves,
            "stem": stem,
            "roots": roots,
            "litter2soil": litter2soil,
            "leaves_respiration_loss": leaves_resp_loss,
            "stem_respiration_loss": stem_resp_loss,
            "roots_respiration_loss": roots_resp_loss,
            "leaf_area_index": leaf_area_index,
            "npp": npp_out,
            "cue": cue,
            "disturbance": disturbance,
        }
