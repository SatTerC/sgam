"""
Plant Functional Type (PFT) definitions for SGAM.

This module defines the PFT enum and default parameters for different plant types
used in the Static Game-theoretic Allocation Model.

Classes:
    PlantFunctionalType: Enum representing different plant functional types.
    PftParams: Dataclass containing physiological parameters for a PFT.
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class PlantFunctionalType(StrEnum):
    """Plant Functional Type (PFT) classification.

    Represents different categories of plants based on their growth form
    and ecological characteristics. Each PFT has associated physiological
    parameters that define carbon allocation and turnover rates.

    Attributes:
        TREE: Woody perennial plants with a single main stem or trunk.
        GRASS: Herbaceous plants with narrow leaves and short lifespan.
        SHRUB: Woody perennial plants with multiple stems, smaller than trees.
        CROP: Cultivated plants grown for agricultural purposes.
    """

    TREE = "tree"
    GRASS = "grass"
    SHRUB = "shrub"
    CROP = "crop"


@dataclass(frozen=True)
class PftParams:
    """Physiological parameters for a Plant Functional Type.

    This dataclass defines carbon allocation fractions and turnover rates
    for different plant tissues (leaf, stem, root). These parameters are
    used in the SGAM to model plant growth and competition dynamics.

    Attributes:
        leaf_base_allocation: Fraction of new carbon allocated to leaves
            during growth. Represents the leaf mass fraction (LMF).
        stem_base_allocation: Fraction of new carbon allocated to stems
            during growth. Represents the stem mass fraction (SMF).
        root_base_allocation: Fraction of new carbon allocated to roots
            during growth. Represents the root mass fraction (RMF).
        leaf_turnover_rate: Fraction of leaf biomass replaced per week (weeks^-1^).
        stem_turnover_rate: Fraction of stem biomass replaced per week (weeks^-1^).
        root_tunover_rate: Fraction of root biomass replaced per week (weeks^-1^).
        leaf_maint_coeff: Fraction of leaf carbon respired per week (weeks^-1^)
            for maintenance metabolism. This is the highest cost across all
            PFTs as leaves require constant enzyme maintenance. Crops have
            the highest values (0.14) meaning 14% of leaf carbon is burned
            weekly without GPP.
        stem_maint_coeff: Fraction of stem carbon respired per week (weeks^-1^)
            for maintenance metabolism. Lower for trees and shrubs due to
            lignified, metabolically inactive wood; higher for grasses and
            crops with succulent, active stems.
        root_maint_coeff: Fraction of root carbon respired per week (weeks^-1^)
            for maintenance metabolism. Reflects the metabolic cost of
            maintaining fine root systems for water and nutrient uptake.
        lue_max: Maximum light use efficiency (gC MJ^-1^). Represents the
            maximum rate of carbon gain per unit of absorbed light. Crops
            have the highest values due to C4 photosynthesis and optimized
            agricultural conditions; shrubs have the lowest, reflecting
            adaptation to resource-poor environments.
        iwue_max: Maximum intrinsic water use efficiency (μmol mol^-1^).
            Represents the maximum ratio of photosynthesis to stomatal
            conductance. Shrubs have the highest values to maintain
            photosynthesis under extreme water tension.
        leaf_carbon_area: Specific leaf area expressed as carbon content
            per unit leaf area (gC/m²).
        wilting_point: Soil moisture (fraction, 0.0-1.0) at which plant
            water stress reaches maximum (f_sm = 0).
        field_capacity: Soil moisture (fraction, 0.0-1.0) at which plant
            water stress is minimum (f_sm = 1).
        vpd_threshold: Vapor Pressure Deficit (Pa) above which stomatal
            conductance begins to decline.
        vpd_sensitivity: Rate of decline in stomatal conductance with
            increasing VPD (Pa^-1^). Used in f_vpd = exp(-gamma * (VPD - threshold)).
    """

    def __post_init__(self) -> None:
        total = (
            self.leaf_base_allocation
            + self.stem_base_allocation
            + self.root_base_allocation
        )
        if not np.isclose(total, 1.0):
            raise ValueError(
                f"Base allocations must sum to 1.0, got {total}: "
                f"leaf={self.leaf_base_allocation}, "
                f"stem={self.stem_base_allocation}, "
                f"root={self.root_base_allocation}"
            )

    # Base allocation fractions (must sum to 1.0)
    leaf_base_allocation: float
    stem_base_allocation: float
    root_base_allocation: float

    # Weekly turnover rates (fraction of pool lost per week)
    leaf_turnover_rate: float
    stem_turnover_rate: float
    root_turnover_rate: float

    # Weekly maintenance coefficients (metabolic cost per unit biomass)
    leaf_maint_coeff: float
    stem_maint_coeff: float
    root_maint_coeff: float

    # Efficiency thresholds
    lue_max: float  # gC / MJ
    iwue_max: float  # μmol / mol

    # Disturbance calibration
    disturbance_threshold: float
    disturbance_leaf_loss_frac: float

    # Physical constants
    leaf_carbon_area: float

    # Hydro-physiological constants
    wilting_point: float
    field_capacity: float
    vpd_threshold: float
    vpd_sensitivity: float


_PFT_PARAMS = {
    PlantFunctionalType.TREE: PftParams(
        leaf_base_allocation=0.25,
        stem_base_allocation=0.45,
        root_base_allocation=0.30,
        leaf_turnover_rate=0.012,
        stem_turnover_rate=0.0002,
        root_turnover_rate=0.010,
        leaf_maint_coeff=0.06,
        stem_maint_coeff=0.005,
        root_maint_coeff=0.03,
        lue_max=2.5,
        iwue_max=450.0,
        disturbance_threshold=0.3,
        disturbance_leaf_loss_frac=0.4,
        leaf_carbon_area=60.0,
        wilting_point=0.12,
        field_capacity=0.35,
        vpd_threshold=800.0,
        vpd_sensitivity=0.0005,
    ),
    PlantFunctionalType.GRASS: PftParams(
        leaf_base_allocation=0.45,
        stem_base_allocation=0.10,
        root_base_allocation=0.45,
        leaf_turnover_rate=0.035,
        stem_turnover_rate=0.015,
        root_turnover_rate=0.025,
        leaf_maint_coeff=0.09,
        stem_maint_coeff=0.03,
        root_maint_coeff=0.05,
        lue_max=3.0,
        iwue_max=350.0,
        disturbance_threshold=0.2,
        disturbance_leaf_loss_frac=0.9,
        leaf_carbon_area=40.0,
        wilting_point=0.08,
        field_capacity=0.30,
        vpd_threshold=500.0,
        vpd_sensitivity=0.0008,
    ),
    PlantFunctionalType.SHRUB: PftParams(
        leaf_base_allocation=0.20,
        stem_base_allocation=0.40,
        root_base_allocation=0.40,
        leaf_turnover_rate=0.010,
        stem_turnover_rate=0.002,
        root_turnover_rate=0.010,
        leaf_maint_coeff=0.07,
        stem_maint_coeff=0.01,
        root_maint_coeff=0.04,
        lue_max=2.2,
        iwue_max=650.0,
        disturbance_threshold=0.25,
        disturbance_leaf_loss_frac=0.5,
        leaf_carbon_area=80.0,
        wilting_point=0.05,
        field_capacity=0.25,
        vpd_threshold=1200.0,
        vpd_sensitivity=0.0003,
    ),
    PlantFunctionalType.CROP: PftParams(
        leaf_base_allocation=0.40,
        stem_base_allocation=0.40,
        root_base_allocation=0.20,
        leaf_turnover_rate=0.050,
        stem_turnover_rate=0.025,
        root_turnover_rate=0.030,
        leaf_maint_coeff=0.12,
        stem_maint_coeff=0.05,
        root_maint_coeff=0.07,
        lue_max=4.2,
        iwue_max=300.0,
        disturbance_threshold=0.1,
        disturbance_leaf_loss_frac=1.0,
        leaf_carbon_area=35.0,
        wilting_point=0.15,
        field_capacity=0.40,
        vpd_threshold=400.0,
        vpd_sensitivity=0.0012,
    ),
}


def get_default_pft_params(pft: PlantFunctionalType) -> PftParams:
    """Get the default physiological parameters for a Plant Functional Type.

    Retrieves the default physiological parameters associated with the given
    plant functional type. These parameters define carbon allocation fractions
    and turnover rates for different plant tissues.

    Args:
        pft: The plant functional type to get parameters for.

    Returns:
        The default physiological parameters for the specified PFT.

    Raises:
        KeyError: If the given PFT is not recognized.

    See Also:
        PftParams: The parameter dataclass definition.
        PlantFunctionalType: Available PFT classifications.

    Example:
        >>> params = get_default_pft_params(PlantFunctionalType.TREE)
        >>> print(params.leaf_base_allocation)
        0.05
    """
    return _PFT_PARAMS[pft]
