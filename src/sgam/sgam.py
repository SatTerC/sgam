"""
SGAM (Simplified Growth/GPP Allocation Model) component.

This module provides the SgamComponent class, which simulates the allocation
of gross primary productivity (GPP) to plant carbon pools (leaf_pool_size, stem_pool_size, root_pool_size)
across different plant types (tree, grass, crop, shrub), including turnover,
respiration, and disturbance events.
"""

import numpy as np
from numpy.typing import NDArray

from .pft import PftParams, PlantFunctionalType, get_default_pft_params


class SgamComponent:
    """The Simplified Growth/GPP Allocation Model (SGAM).

    Simulates the allocation of gross primary productivity (GPP) to plant
    carbon pools (leaf_pool_size, stem_pool_size, root_pool_size) for
    4 plant types (tree, grass, crop, shrub) over time, based on environmental
    drivers and physiological parameters. It accounts for dynamic allocation,
    turnover, respiration, disturbance/harvest events, and outputs pool sizes
    and fluxes.

    Args:
        plant_type: Type of plant (tree, grass, crop, or shrub).
        pft_params: Optional custom PFT parameters. If None, uses defaults
            for the specified plant_type.
        use_dynamic_allocation: If True (default), allocation fractions
            vary with environmental conditions (temperature, moisture, VPD).
            If False, use fixed base allocations from pft_params.

    Todo:
        - Refine crop modelling --> growing_season_limit necessary ?
        - Add PC output for RothC when crop harvested or not emerged
        - Add grazing -> manure return to RothC
    """

    def __init__(
        self,
        plant_type: PlantFunctionalType,
        pft_params: PftParams | None = None,
        use_dynamic_allocation: bool = True,
    ):
        self.plant_type = plant_type
        self.pft_params = (
            pft_params if pft_params is not None else get_default_pft_params(plant_type)
        )
        self.use_dynamic_allocation = use_dynamic_allocation

    def compute_cue(self, lue: NDArray, iwue: NDArray) -> NDArray:
        """Compute carbon use efficiency (CUE) from light use efficiency and
        intrinsic water use efficiency.

        CUE = CUE_{max} . f(LUE_{norm}) . f(IWUE_{norm})

        Args:
            lue: Weekly mean light use efficiency (gC/MJ).
            iwue: Weekly mean intrinsic water use efficiency (umol/mol).

        Returns:
            Carbon use efficiency values.
        """
        # Scale LUE and IWUE against theoretical maximums
        # np.clip ensures values stay between 0 and 1 regardless of "outlier" days
        lue_score = np.clip(lue / self.pft_params.lue_max, 0, 1)
        iwue_score = np.clip(iwue / self.pft_params.iwue_max, 0, 1)

        # CUE is highest when the plant is not limited by light or water stress
        lue_iwue_score_avg = (lue_score + iwue_score) / 2

        # Base CUE of 0.2 (high stress) to 0.7 (optimal growth)
        cue = 0.2 + lue_iwue_score_avg * (0.7 - 0.2)

        return cue

    def _compute_drought_modifier(
        self,
        soil_moisture: NDArray,
        vpd: NDArray,
    ) -> NDArray:
        """Compute drought modifier based on soil moisture and VPD.

        Args:
            soil_moisture: Soil moisture values.
            vpd: Vapor pressure deficit values (Pa).

        Returns:
            Drought modifier values.
        """
        # NOTE: This is the old version. See non-underscored update below.
        #
        # Calculate environmental stress thresholds from percentiles of input data
        # 25th percentile of soil moisture serves as moisture stress threshold
        # 75th percentile of VPD serves as maximum vapor pressure deficit threshold
        # Clip the value between vaguely physically plausible limits:
        # soil_moisture = 0.05 is around wilting point
        # VPD = 5000 is extreme desert conditions
        moisture_threshold = np.clip(np.percentile(soil_moisture, 25), 0.05, 1.0)
        vpd_max = np.clip(np.percentile(vpd, 75), 100, 5000)

        normalized_moisture = np.minimum(soil_moisture / moisture_threshold, 1.0)
        normalized_vpd = np.minimum(vpd / vpd_max, 1.0)

        return (1 - normalized_moisture) + normalized_vpd

    def compute_drought_modifier(
        self,
        soil_moisture: NDArray,
        vpd: NDArray,
    ) -> NDArray:
        r"""Compute the environmental stress scalar using Liebig's Law of the Minimum.

        This modifier accounts for both edaphic (soil) and atmospheric (VPD) water
        stress. The final modifier is the minimum of the two individual stress
        functions, ranging from 0.0 (maximum stress) to 1.0 (no stress).

        **Soil Moisture Stress** ($f_{sm}$):
        Scales linearly between the wilting point ($\theta_{wp}$) and a
        reference soil moisture ($\theta_{ref}$):

        $$
        f_{sm} = \begin{cases}
        0 & \theta < \theta_{wp} \\
        \frac{\theta - \theta_{wp}}{\theta_{ref} - \theta_{wp}} & \theta_{wp} \le \theta \le \theta_{ref} \\
        1 & \theta > \theta_{ref}
        \end{cases}
        $$

        **VPD Stress** ($f_{vpd}$):
        Represents stomatal closure as a function of vapor pressure deficit
        using an exponential decay:

        $$
        f_{vpd} = \exp(-\gamma \cdot \max(0, VPD - VPD_{threshold}))
        $$

        Args:
            soil_moisture: Volumetric soil water content ($m^3/m^3$).
            vpd: Vapor Pressure Deficit in Pascals (Pa).

        Returns:
            The combined drought modifier $\min(f_{sm}, f_{vpd})$.
        """
        # Retrieve PFT-specific sensitivities
        theta_wp = self.pft_params.wilting_point
        theta_ref = self.pft_params.field_capacity
        gamma = self.pft_params.vpd_sensitivity
        vpd_thresh = self.pft_params.vpd_threshold

        f_sm = np.clip((soil_moisture - theta_wp) / (theta_ref - theta_wp), 0.0, 1.0)

        # 2. VPD Stress (Exponential decay)
        # gamma defines sensitivity. 0.0005 is a standard value for kPa-based Pa.
        # If VPD is in Pa, gamma should be around 0.00005 to 0.0001
        gamma = 0.0001
        f_vpd = np.exp(
            -gamma * np.maximum(vpd - vpd_thresh, 0.0)
        )  # Threshold of 500Pa before stress starts

        # 3. Combine using the Minimum (Liebig's Law)
        # The most limiting factor dominates the plant's physiology
        return np.minimum(f_sm, f_vpd)

    def compute_allocation_fractions(
        self,
        temperature: NDArray[np.float64],
        soil_moisture: NDArray[np.float64],
        vpd: NDArray[np.float64],
        week_of_year: NDArray[np.float64],
        use_dynamic_allocation: bool | None = None,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Compute carbon allocation fractions for leaf, stem, and root pools.

        Args:
            temperature: Weekly mean air temperature (degC).
            soil_moisture: Weekly mean soil moisture content (normalized or mm).
            vpd: Weekly mean vapor pressure deficit (Pa).
            week_of_year: The week number of the simulation year (1 to 52).
            use_dynamic_allocation: If True, allocation varies with environmental
                conditions. If False, returns normalized base allocations.
                Defaults to self.use_dynamic_allocation.

        Returns:
            Allocation fractions for (leaf, stem, root), summing to 1.0.
        """
        if use_dynamic_allocation is None:
            use_dynamic_allocation = self.use_dynamic_allocation

        if not use_dynamic_allocation:
            # NOTE: if falling back on non-dynamic allocations we don't need any input
            # data, but I don't know if it's worth bothering to actually implement this;
            # static allocations should be a sanity check rather than a supported option.
            leaf = np.full_like(temperature, self.pft_params.leaf_base_allocation)
            stem = np.full_like(temperature, self.pft_params.stem_base_allocation)
            root = np.full_like(temperature, self.pft_params.root_base_allocation)
            total = leaf + stem + root
            return (leaf / total, stem / total, root / total)

        # 1. Seasonality Modifier
        # Peak allocation to leaves occurs around the summer solstice (Week 26).
        # Shifted by 12 weeks so the sine wave begins its climb in spring.
        seasonality_mod = 0.15 * np.sin(2 * np.pi * (week_of_year - 12) / 52.0)

        # 2. Temperature Modifier
        # Linear shift favoring leaf allocation in warmer weeks, up to a 10% swing.
        temperature_mod = np.clip((temperature - 20) / 40, -0.1, 0.1)

        # 3. Drought Stress Modifier
        # moisture_stress: 0.0 (wet) to 1.0 (dry).
        # Assumes 0.5 is a generic PFT-agnostic moisture midpoint.
        moisture_stress = np.clip(1.0 - (soil_moisture / 0.5), 0, 1)

        # VPD stress: scaled against the PFT's specific IWUE threshold.
        vpd_stress = np.clip(vpd / self.pft_params.iwue_max, 0, 1)

        # Total drought bonus shifts up to 15% extra carbon to the roots for foraging.
        drought_root_bonus = 0.15 * np.maximum(moisture_stress, vpd_stress)

        # 4. Apply Modifiers to PFT Base Values
        # We maintain a biological floor of 5% for leaves/roots and 1% for stems.
        dynamic_leaf = np.maximum(
            0.05,
            self.pft_params.leaf_base_allocation
            + seasonality_mod
            + temperature_mod
            - (drought_root_bonus * 0.5),
        )

        dynamic_stem = np.maximum(
            0.01, self.pft_params.stem_base_allocation - (drought_root_bonus * 0.5)
        )

        dynamic_root = np.maximum(
            0.05,
            self.pft_params.root_base_allocation
            - seasonality_mod
            - temperature_mod
            + drought_root_bonus,
        )

        # 5. Final Normalization
        total_allocation = dynamic_leaf + dynamic_stem + dynamic_root

        return (
            dynamic_leaf / total_allocation,
            dynamic_stem / total_allocation,
            dynamic_root / total_allocation,
        )

    def partition_incoming_gpp(
        self,
        gpp: NDArray[np.float64],
        temperature: NDArray[np.float64],
        soil_moisture: NDArray[np.float64],
        vpd: NDArray[np.float64],
        lue: NDArray[np.float64],
        iwue: NDArray[np.float64],
        week_of_year: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # Compute dynamic multipliers
        carbon_use_efficiency = self.compute_cue(lue, iwue)

        (
            leaf_allocation_fraction,
            stem_allocation_fraction,
            root_allocation_fraction,
        ) = self.compute_allocation_fractions(
            temperature,
            soil_moisture,
            vpd,
            week_of_year,
        )

        # Partition incoming GPP between leaf, stem, root pools
        leaf_gpp = gpp * leaf_allocation_fraction
        stem_gpp = gpp * stem_allocation_fraction
        root_gpp = gpp * root_allocation_fraction

        # Partition each pool's GPP between *potential* growth (NPP) and autotrophic respiration (Ra)
        leaf_respiration = leaf_gpp * (1 - carbon_use_efficiency)
        stem_respiration = stem_gpp * (1 - carbon_use_efficiency)
        root_respiration = root_gpp * (1 - carbon_use_efficiency)
        leaf_npp = leaf_gpp * carbon_use_efficiency
        stem_npp = stem_gpp * carbon_use_efficiency
        root_npp = root_gpp * carbon_use_efficiency

        return (
            np.stack([leaf_respiration, stem_respiration, root_respiration]),
            np.stack([leaf_npp, stem_npp, root_npp]),
        )

    def _compute_disturbance_deltas(
        self,
        leaf_pool_size: float,
        stem_pool_size: float,
        root_pool_size: float,
        disturbance_severity: float,
    ) -> tuple[float, float, float, float, float]:
        if self.plant_type == PlantFunctionalType.CROP:
            # Any severity > 0 is treated as a clear cut / harvest.
            # Hence, pools are zeroed are disturbance loss = pool size.
            Δ_leaf = -leaf_pool_size
            Δ_stem = -stem_pool_size
            Δ_root = -root_pool_size

            # Leaf and stem carbon are assumed to have been removed;
            # only root carbon joins the litter pool.
            Δ_litter = root_pool_size
            Δ_removed = leaf_pool_size + stem_pool_size

        else:
            # TODO: this only models grazing (I think) - needs work!
            impact_frac = (
                disturbance_severity * self.pft_params.disturbance_leaf_loss_frac
            )
            leaf_disturbance_loss = leaf_pool_size * impact_frac

            Δ_leaf = -leaf_disturbance_loss
            Δ_litter = leaf_disturbance_loss
            Δ_stem, Δ_root, Δ_removed = 0.0, 0.0, 0.0

        return (Δ_leaf, Δ_stem, Δ_root, Δ_litter, Δ_removed)

    def _forward(
        self,
        leaf_pool_init: float,
        stem_pool_init: float,
        root_pool_init: float,
        leaf_npp: NDArray[np.float64],
        stem_npp: NDArray[np.float64],
        root_npp: NDArray[np.float64],
        disturbance_severity: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        leaf_pool_series = []
        stem_pool_series = []
        root_pool_series = []
        litter_pool_series = []
        removed_series = []

        leaf_pool = leaf_pool_init
        stem_pool = stem_pool_init
        root_pool = root_pool_init
        litter_pool = 0.0
        removed = 0.0

        n_weeks = len(leaf_npp)
        for t in range(n_weeks):
            # Natural turnover (litterfall) is proportional to pool size
            # (constant decay rate -> exponential decay)
            leaf_turnover = leaf_pool * self.pft_params.leaf_turnover_rate
            stem_turnover = stem_pool * self.pft_params.stem_turnover_rate
            root_turnover = root_pool * self.pft_params.root_turnover_rate

            # Compute "natural" mass update, i.e. in the absence of disturbances
            Δ_leaf = leaf_npp[t] - leaf_turnover
            Δ_stem = stem_npp[t] - stem_turnover
            Δ_root = root_npp[t] - root_turnover
            Δ_litter = leaf_turnover + stem_turnover + root_turnover

            leaf_pool += Δ_leaf
            stem_pool += Δ_stem
            root_pool += Δ_root
            litter_pool += Δ_litter

            if disturbance_severity[t] > 0:
                Δ_leaf, Δ_stem, Δ_root, Δ_litter, Δ_removed = (
                    self._compute_disturbance_deltas(
                        leaf_pool_size=leaf_pool,
                        stem_pool_size=stem_pool,
                        root_pool_size=root_pool,
                        disturbance_severity=disturbance_severity[t],
                    )
                )
                leaf_pool += Δ_leaf
                stem_pool += Δ_stem
                root_pool += Δ_root
                litter_pool += Δ_litter
                removed += Δ_removed

            leaf_pool_series.append(leaf_pool)
            stem_pool_series.append(stem_pool)
            root_pool_series.append(root_pool)
            litter_pool_series.append(litter_pool)
            removed_series.append(removed)

        return np.stack(
            [
                leaf_pool_series,
                stem_pool_series,
                root_pool_series,
                litter_pool_series,
                removed_series,
            ]
        )

    def forward(
        self,
        gpp: NDArray[np.float64],
        temperature: NDArray[np.float64],
        soil_moisture: NDArray[np.float64],
        vpd: NDArray[np.float64],
        lue: NDArray[np.float64],
        iwue: NDArray[np.float64],
        week_of_year: NDArray[np.float64],
        disturbances: NDArray[np.float64],
        leaf_pool_init: float,
        stem_pool_init: float,
        root_pool_init: float,
    ) -> dict[str, NDArray]:
        """Simulate weekly plant growth and carbon allocation using a mass-balance approach.

        Args:
            gpp: Weekly total gross primary productivity (gC).
            temperature: Weekly mean air temperature (degC).
            soil_moisture: Weekly mean soil moisture content (normalized or mm).
            vpd: Weekly mean vapor pressure deficit (Pa).
            lue: Weekly mean light use efficiency (gC/MJ).
            iwue: Weekly mean intrinsic water use efficiency (umol/mol).
            week_of_year: Weekly timestep index of the year (1-52).
            disturbances: The maximum daily relative decline (0.0 to 1.0) observed
                during the week. Values of 0.0 indicate no disturbance event.
            leaf_pool_init: Initial leaf biomass pool size (gC).
            stem_pool_init: Initial stem biomass pool size (gC).
            root_pool_init: Initial root biomass pool size (gC).

        Returns:
            Carbon pools (gC), fluxes (gC), and diagnostics (LAI, NPP, CUE).
        """

        respiration, npp = self.partition_incoming_gpp(
            gpp=gpp,
            temperature=temperature,
            soil_moisture=soil_moisture,
            vpd=vpd,
            lue=lue,
            iwue=iwue,
            week_of_year=week_of_year,
        )

        leaf_npp, stem_npp, root_npp = npp

        # Run the forward model, iteratively updating the pool sizes
        pools = self._forward(
            leaf_pool_init=leaf_pool_init,
            stem_pool_init=stem_pool_init,
            root_pool_init=root_pool_init,
            leaf_npp=leaf_npp,
            stem_npp=stem_npp,
            root_npp=root_npp,
            disturbance_severity=disturbances,
        )

        leaf_pool, stem_pool, root_pool, litter_pool, removed = pools

        # TODO: decide whether to re-compute turnover and disturbance deltas here,
        # or accumulate them in _forward. They probably should be returned alongside pools.

        return dict(
            leaf_pool=leaf_pool,
            stem_pool=stem_pool,
            root_pool=root_pool,
            litter_pool=litter_pool,
            removed=removed,
        )

    def __call__(
        self,
        gpp: NDArray[np.float64],
        temperature: NDArray[np.float64],
        soil_moisture: NDArray[np.float64],
        vpd: NDArray[np.float64],
        lue: NDArray[np.float64],
        iwue: NDArray[np.float64],
        week_of_year: NDArray[np.float64],
        disturbances: NDArray[np.float64],
        leaf_pool_init: float,
        stem_pool_init: float,
        root_pool_init: float,
    ) -> dict[str, NDArray]:
        """Alias for ``forward``.

        Args:
            gpp: Weekly total gross primary productivity (gC).
            temperature: Weekly mean air temperature (degC).
            soil_moisture: Weekly mean soil moisture content (normalized or mm).
            vpd: Weekly mean vapor pressure deficit (Pa).
            lue: Weekly mean light use efficiency (gC/MJ).
            iwue: Weekly mean intrinsic water use efficiency (umol/mol).
            week_of_year: Weekly timestep index of the year (1-52).
            disturbances: The maximum daily relative decline (0.0 to 1.0) observed
                during the week.
            leaf_pool_init: Initial leaf biomass pool size (gC).
            stem_pool_init: Initial stem biomass pool size (gC).
            root_pool_init: Initial root biomass pool size (gC).

        Returns:
            Carbon pools (gC), fluxes (gC), and diagnostics (LAI, NPP, CUE).
        """
        return self.forward(
            gpp=gpp,
            temperature=temperature,
            soil_moisture=soil_moisture,
            vpd=vpd,
            lue=lue,
            iwue=iwue,
            week_of_year=week_of_year,
            disturbances=disturbances,
            leaf_pool_init=leaf_pool_init,
            stem_pool_init=stem_pool_init,
            root_pool_init=root_pool_init,
        )
