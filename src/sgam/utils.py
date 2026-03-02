"""
Utility functions for SGAM (Simplified Growth/GPP Allocation Model).

This module contains helper functions for carbon pool calculations,
allocation computations, and related operations.
"""

import numpy as np
from numpy.typing import NDArray


def rescale_to_unit_interval(x: NDArray) -> NDArray:
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


def compute_relative_changes(x: NDArray) -> NDArray:
    """
    Compute relative changes between consecutive timesteps.

    Parameters
    ----------
    x : NDArray
        Input array.

    Returns
    -------
    NDArray
        Relative changes, with first element as 0.
    """
    Δx = np.zeros_like(x)
    Δx[1:] = (x[1:] - x[:-1]) / np.maximum(x[:-1], 1e-6)
    return Δx


def solve_recurrence(initial_value: float, a: NDArray, b: NDArray) -> NDArray:
    """
    Solve first-order linear recurrence: x[i+1] = a[i] * x[i] + b[i].

    The recurrence expands to:
    x[i] = x[0] * prod(a[0:i-1]) + sum(b[j] * prod(a[j+1:i]) for j in range(i))

    for i = 1, 2, ..., m-1, where x[0] = initial_value.

    This can be computed using cumulative products. We use the identity:
    sum(b[j] * prod(a[j+1:i]) for j in range(i))
    = sum(b[j] * cumprod_a[i] / cumprod_a[j+1] for j in range(i))

    where cumprod_a[i] = prod(a[0:i]) with cumprod_a[0] = 1.

    Parameters
    ----------
    initial_value : float
        Initial value at start of sequence.
    a : NDArray
        Coefficients where x[i+1] = a[i] * x[i] + b[i].
    b : NDArray
        Additive terms.

    Returns
    -------
    NDArray
        Values for each timestep (length m).
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

    pools = initial_value * cumprod_a[1:] + weighted_sum

    return pools


def find_segments(mask: NDArray) -> list[tuple[int, int]]:
    """
    Find start and end indices of contiguous segments where mask is False.

    Each segment is extended by 1 to include the following True index (if any),
    allowing pool values to be computed on disturbance days before the disturbance
    loss is applied in a subsequent step.

    Parameters
    ----------
    mask : NDArray
        Boolean array. Segments are defined as regions where mask is False.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) tuples for each segment.
    """
    n_timesteps = len(mask)
    segments = []

    true_indices = np.where(mask)[0]

    if len(true_indices) == 0:
        return [(0, n_timesteps)]

    # Segment from start (index 0) to first True value (extended by 1)
    if true_indices[0] > 0:
        segments.append((0, true_indices[0] + 1))

    # Segments between consecutive True values (each extended by 1)
    for i in range(len(true_indices) - 1):
        start = true_indices[i] + 1
        end = true_indices[i + 1] + 1  # extend to include the True index
        if start < end:
            segments.append((start, end))

    # Segment from last True value to end (extended by 1)
    if true_indices[-1] < n_timesteps - 1:
        segments.append((true_indices[-1] + 1, n_timesteps))

    return segments
