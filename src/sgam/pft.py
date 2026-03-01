"""
Plant Functional Type (PFT) definitions for SGAM.

This module defines the PFT enum and default parameters for different plant types.
"""

from dataclasses import dataclass
from enum import StrEnum


class PlantFunctionalType(StrEnum):
    TREE = "tree"
    GRASS = "grass"
    SHRUB = "shrub"
    CROP = "crop"


@dataclass(frozen=True)
class PftParams:
    leaf_base_allocation: float
    stem_base_allocation: float
    root_base_allocation: float
    leaf_turnover_rate: float
    stem_turnover_rate: float
    root_turnover_rate: float
    leaf_carbon_area: float = 30.0
    disturbance_limit: float = 0.3
    growing_season_limit: float = 10.0


PFT_PARAMS: dict[PlantFunctionalType, PftParams] = {
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
