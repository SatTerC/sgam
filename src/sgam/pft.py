"""
Plant Functional Type (PFT) definitions for SGAM.

This module defines the PFT enum and default parameters for different plant types
used in the Static Game-theoretic Allocation Model.

Functions:
    get_default_pft_params: Get the default physiological parameters for a Plant Functional Type.

Classes:
    PlantFunctionalType: Enum representing different plant functional types.
    PftParams: Dataclass containing physiological parameters for a PFT.
"""

from dataclasses import dataclass
from enum import StrEnum
from importlib.resources import files

import numpy as np
import tomllib


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
        root_turnover_rate: Fraction of root biomass replaced per week (weeks^-1^).
        lue_max: Maximum light use efficiency (gC MJ^-1^). Represents the
            maximum rate of carbon gain per unit of absorbed light. Crops
            have the highest values due to C4 photosynthesis and optimized
            agricultural conditions; shrubs have the lowest, reflecting
            adaptation to resource-poor environments.
        iwue_max: Maximum intrinsic water use efficiency (μmol mol^-1^).
            Represents the maximum ratio of photosynthesis to stomatal
            conductance. Shrubs have the highest values to maintain
            photosynthesis under extreme water tension.
        disturbance_threshold: Threshold for disturbance detection (fraction).
        disturbance_leaf_loss_frac: Fraction of leaf pool lost per unit
            disturbance severity.
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


_PFT_PARAMS: dict[PlantFunctionalType, PftParams] | None = None


def _load_pft_params() -> dict[PlantFunctionalType, PftParams]:
    """Load PFT parameters from the default TOML configuration file."""
    tomllib_data = tomllib.loads(
        files("sgam.config").joinpath("pft_defaults.toml").read_text()
    )

    params = {}
    for pft in PlantFunctionalType:
        section = tomllib_data[pft.value]
        params[pft] = PftParams(
            leaf_base_allocation=section["leaf_base_allocation"],
            stem_base_allocation=section["stem_base_allocation"],
            root_base_allocation=section["root_base_allocation"],
            leaf_turnover_rate=section["leaf_turnover_rate"],
            stem_turnover_rate=section["stem_turnover_rate"],
            root_turnover_rate=section["root_turnover_rate"],
            lue_max=section["lue_max"],
            iwue_max=section["iwue_max"],
            disturbance_threshold=section["disturbance_threshold"],
            disturbance_leaf_loss_frac=section["disturbance_leaf_loss_frac"],
            leaf_carbon_area=section["leaf_carbon_area"],
            wilting_point=section["wilting_point"],
            field_capacity=section["field_capacity"],
            vpd_threshold=section["vpd_threshold"],
            vpd_sensitivity=section["vpd_sensitivity"],
        )
    return params


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
    global _PFT_PARAMS
    if _PFT_PARAMS is None:
        _PFT_PARAMS = _load_pft_params()
    return _PFT_PARAMS[pft]
