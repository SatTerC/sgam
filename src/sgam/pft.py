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
        leaf_turnover_rate: Annual leaf turnover rate (1/year). Fraction
            of leaf biomass replaced per year.
        stem_turnover_rate: Annual stem turnover rate (1/year). Fraction
            of stem biomass replaced per year.
        leaf_carbon_area: Specific leaf area expressed as carbon content
            per unit leaf area (gC/m²). Default: 30.0.
        disturbance_limit: Maximum fraction of biomass that can be removed
            by disturbance events. Default: 0.3.
        growing_season_limit: Minimum number of frost-free days required
            for successful growth. Default: 10.0.
    """

    leaf_base_allocation: float
    stem_base_allocation: float
    root_base_allocation: float
    leaf_turnover_rate: float
    stem_turnover_rate: float
    root_turnover_rate: float
    leaf_carbon_area: float = 30.0
    disturbance_limit: float = 0.3
    growing_season_limit: float = 10.0


def get_default_pft_params(pft: PlantFunctionalType) -> PftParams:
    """Get the default physiological parameters for a Plant Functional Type.

    Retrieves the default physiological parameters associated with the given
    plant functional type. These parameters define carbon allocation fractions
    and turnover rates for different plant tissues.

    Args:
        pft: The plant functional type to get parameters for.

    Returns:
        PftParams: The default physiological parameters for the specified PFT.

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


_PFT_PARAMS: dict[PlantFunctionalType, PftParams] = {
    PlantFunctionalType.TREE: PftParams(
        leaf_base_allocation=0.05,
        stem_base_allocation=0.65,
        root_base_allocation=0.30,
        leaf_turnover_rate=0.01,
        stem_turnover_rate=0.0001,
        root_turnover_rate=0.005,
    ),
    PlantFunctionalType.GRASS: PftParams(
        leaf_base_allocation=0.40,
        stem_base_allocation=0.10,
        root_base_allocation=0.50,
        leaf_turnover_rate=0.01,
        stem_turnover_rate=0.001,
        root_turnover_rate=0.005,
    ),
    PlantFunctionalType.SHRUB: PftParams(
        leaf_base_allocation=0.10,
        stem_base_allocation=0.40,
        root_base_allocation=0.50,
        leaf_turnover_rate=0.01,
        stem_turnover_rate=0.0005,
        root_turnover_rate=0.005,
    ),
    PlantFunctionalType.CROP: PftParams(
        leaf_base_allocation=0.25,
        stem_base_allocation=0.50,
        root_base_allocation=0.25,
        leaf_turnover_rate=0.01,
        stem_turnover_rate=0.001,
        root_turnover_rate=0.005,
    ),
}
