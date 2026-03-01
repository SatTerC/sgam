"""
Utility functions for SGAM (Simplified Growth/GPP Allocation Model).

This module contains helper functions for carbon pool calculations,
allocation computations, and related operations.
"""

import numpy as np
from numpy.typing import NDArray


def _rescale_to_unit_interval(x: NDArray) -> NDArray:
    """
    Normalize array to [0, 1] range using min-max normalization.

    Parameters
    ----------
    x : NDArray
        Input array.

    Returns
    -------
    NDArray
        Normalized array.
    """
    min_x = np.nanmin(x)
    max_x = np.nanmax(x)
    if max_x > min_x:
        return (x - min_x) / (max_x - min_x)
    else:
        return np.zeros_like(x)


def compute_cue(lue: NDArray, iwue: NDArray) -> NDArray:
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
    lue_norm = _rescale_to_unit_interval(lue)
    iwue_norm = _rescale_to_unit_interval(iwue)
    iwue_norm_inv = 1 - iwue_norm
    cue_raw = 0.5 * (lue_norm + iwue_norm_inv)
    return 0.2 + cue_raw * (0.9 - 0.2)


def compute_growing_season(temp: NDArray, threshold: float) -> NDArray:
    """
    Compute boolean mask for growing season based on temperature threshold.

    Parameters
    ----------
    temp : NDArray
        Temperature values (degC).
    threshold : float
        Minimum temperature for growing season (degC).

    Returns
    -------
    NDArray
        Boolean array indicating growing season.
    """
    return temp > threshold


def compute_relative_changes(values: NDArray) -> NDArray:
    """
    Compute relative changes between consecutive timesteps.

    Parameters
    ----------
    values : NDArray
        Input values.

    Returns
    -------
    NDArray
        Relative changes, with first element as 0.
    """
    result = np.zeros_like(values)
    result[1:] = (values[1:] - values[:-1]) / np.maximum(values[:-1], 1e-6)
    return result


def compute_drought_modifier(
    soil_moisture: NDArray, vpd: NDArray, moisture_threshold: float, vpd_max: float
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


def get_allocation_bases(plant_type: str) -> tuple[float, float, float]:
    """
    Get base allocation percentages for a plant type.

    Parameters
    ----------
    plant_type : str
        Type of plant ('tree', 'grass', 'crop', or 'shrub').

    Returns
    -------
    tuple[float, float, float]
        Base allocation percentages for (leaves, stem, roots).

    Raises
    ------
    ValueError
        If plant type is not supported.
    """
    pt = plant_type.lower()
    if pt == "tree":
        return 0.05, 0.65, 0.30
    elif pt == "grass":
        return 0.40, 0.10, 0.50
    elif pt == "shrub":
        return 0.10, 0.40, 0.50
    elif pt == "crop":
        return 0.25, 0.50, 0.25
    else:
        raise ValueError(f"Unsupported plant type: {plant_type}")


def compute_allocation_percentages(
    temp: NDArray,
    doy: NDArray,
    soil_moisture: NDArray,
    vpd: NDArray,
    ts: float,
    moisture_threshold: float,
    vpd_max: float,
    base_leaves: float,
    base_stem: float,
    base_roots: float,
) -> dict[str, NDArray]:
    """
    Compute dynamic allocation percentages based on environmental factors.

    Parameters
    ----------
    temp : NDArray
        Temperature values (degC).
    doy : NDArray
        Day of year values.
    soil_moisture : NDArray
        Soil moisture values.
    vpd : NDArray
        Vapor pressure deficit values (Pa).
    ts : float
        Timestep in days.
    moisture_threshold : float
        Threshold for soil moisture stress.
    vpd_max : float
        Maximum VPD value.
    base_leaves : float
        Base allocation fraction for leaves.
    base_stem : float
        Base allocation fraction for stem.
    base_roots : float
        Base allocation fraction for roots.

    Returns
    -------
    dict[str, NDArray]
        Dictionary with allocation percentages for 'leaves', 'stem', 'roots'.
    """
    seasonality_mod = np.sin(2 * np.pi * doy / 365.0)
    temp_mod = (temp - 20) / 100

    dynamic_leaves = np.maximum(
        0, base_leaves + 0.15 * seasonality_mod + 0.1 * temp_mod
    )
    dynamic_roots = np.maximum(0, base_roots - 0.15 * seasonality_mod - 0.05 * temp_mod)
    dynamic_stem = np.maximum(0, base_stem - 0.05 * temp_mod)

    total_dynamic = dynamic_leaves + dynamic_stem + dynamic_roots
    total_dynamic = np.maximum(total_dynamic, 1e-10)

    dynamic_leaves = dynamic_leaves / total_dynamic
    dynamic_stem = dynamic_stem / total_dynamic
    dynamic_roots = dynamic_roots / total_dynamic

    drought_modifier = compute_drought_modifier(
        soil_moisture, vpd, moisture_threshold, vpd_max
    )

    root_adjustment = drought_modifier * 0.1
    leaf_stem_adjustment = -drought_modifier * 0.1

    final_roots = np.maximum(0, dynamic_roots + root_adjustment)
    final_leaves = np.maximum(0, dynamic_leaves + leaf_stem_adjustment * 0.7)
    final_stem = np.maximum(0, dynamic_stem + leaf_stem_adjustment * 0.3)

    total_percentage = final_leaves + final_stem + final_roots
    total_percentage = np.maximum(total_percentage, 1e-10)

    return {
        "leaves": final_leaves / total_percentage,
        "stem": final_stem / total_percentage,
        "roots": final_roots / total_percentage,
    }


def solve_pool_recurrence(initial_pool: float, a: NDArray, b: NDArray) -> NDArray:
    """
    Solve pool recurrence: pool[i+1] = a[i] * pool[i] + b[i]

    The recurrence expands to:
    pool[i] = pool[0] * prod(a[0:i-1]) + sum(b[j] * prod(a[j+1:i]) for j in range(i))

    for i = 1, 2, ..., m-1, where pool[0] = initial_pool.

    This can be computed using cumulative products. We use the identity:
    sum(b[j] * prod(a[j+1:i]) for j in range(i))
    = sum(b[j] * cumprod_a[i] / cumprod_a[j+1] for j in range(i))

    where cumprod_a[i] = prod(a[0:i]) with cumprod_a[0] = 1.

    Parameters
    ----------
    initial_pool : float
        Initial pool value at start of epoch.
    a : NDArray
        Coefficients where pool[i+1] = a[i] * pool[i] + b[i].
        Computed as: a = 1 - turnover_factor * litter_cue_modifier / ts.
    b : NDArray
        Net flux (allocated - respiration).

    Returns
    -------
    NDArray
        Pool values for each timestep in the epoch (length m).
    """
    m = len(a)

    cumprod_a = np.ones(m + 1)
    cumprod_a[1:] = np.cumprod(a)

    i_idx, j_idx = np.meshgrid(np.arange(1, m + 1), np.arange(m), indexing="ij")

    valid = i_idx > j_idx

    with np.errstate(divide="ignore", invalid="ignore"):
        prod_values = np.where(valid, cumprod_a[i_idx] / cumprod_a[j_idx + 1], 0.0)

    contributions = b[j_idx] * prod_values

    weighted_sum = np.sum(contributions, axis=1)

    pools = initial_pool * cumprod_a[1:] + weighted_sum

    return pools


def find_epoch_boundaries(disturbance_mask: NDArray) -> list[tuple[int, int]]:
    """
    Find start and end indices of non-disturbance epochs.

    Parameters
    ----------
    disturbance_mask : NDArray
        Boolean array indicating disturbance events.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) tuples for each epoch.
    """
    n = len(disturbance_mask)
    epochs = []

    disturbance_indices = np.where(disturbance_mask)[0]

    if len(disturbance_indices) == 0:
        return [(0, n)]

    if disturbance_indices[0] > 0:
        epochs.append((0, disturbance_indices[0]))

    for i in range(len(disturbance_indices) - 1):
        start = disturbance_indices[i] + 1
        end = disturbance_indices[i + 1]
        if start < end:
            epochs.append((start, end))

    if disturbance_indices[-1] < n - 1:
        epochs.append((disturbance_indices[-1] + 1, n))

    return epochs
