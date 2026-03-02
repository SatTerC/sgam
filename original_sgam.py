"""
This is the original version of SGAM copied from
https://github.com/vmyrgiotis/coupled-ecosystem-carbon-model/blob/v0/src/coupled_ecosystem_carbon_model/scem.py

----------------------------------------------
Simplified Growth/GPP allocation model (SGAM)
----------------------------------------------
"""

import numpy as np


def sgam(
    plant_type,
    df,
    ini_pool,
    leaf_turnover_rate=0.01,
    stem_turnover_rate=0.0001,
    root_turnover_rate=0.005,
    lca=30.0,
    disturbance_limit=0.3,
    growing_season_limit=10,
):
    """
    The Simplified Growth/GPP Allocation Model (SGAM) simulates the allocation of gross primary productivity (GPP)
    to plant carbon pools (leaves, stem, roots) for 4 plant types (tree, grass, crop, shrub) over time,
    based on environmental drivers and physiological parameters.
    It accounts for dynamic allocation, turnover, respiration, disturbance/harvest events, and outputs pool sizes and fluxes.

    Parameters
        ----------
        plant_type : str
            Type of plant ('tree', 'grass', 'crop', or 'shrub').
        df : pandas.DataFrame
            Input dataframe with columns: 'soil_moisture', 'gpp', 'iwue', 'lue', 'temp_degC', 'vpd_Pa', 'lai_obs'.
            The index must be a pandas.DatetimeIndex.
        ini_pool : list or array-like
            Initial carbon pool sizes [leaves, stem, roots].
        leaf_turnover_rate : float, optional
            Daily turnover rate for leaves (default: 0.01).
        stem_turnover_rate : float, optional
            Daily turnover rate for stem (default: 0.0001).
        root_turnover_rate : float, optional
            Daily turnover rate for roots (default: 0.005).
        lca : float, optional
            Leaf carbon area conversion factor (default: 30.0).
        disturbance_limit : float, optional
            Threshold for detecting disturbance/harvest events (default: 0.3).
        growing_season_limit : float, optional
            Minimum temperature (degC) for growing season (default: 7).

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed as input, with columns:
        - 'leaves', 'stem', 'roots': Carbon pool sizes.
        - 'litter2soil': Daily litter carbon to soil.
        - 'leaves_respiration_loss', 'stem_respiration_loss', 'roots_respiration_loss': Daily respiration losses.
        - 'leaf_area_index': Simulated LAI.
        - 'npp': Net primary productivity.
        - 'cue': Carbon use efficiency timeseries.
        - 'disturbance': Carbon loss due to disturbance/harvest.

    Raises
    ------
    ValueError
        If required timeseries ('lue' or 'iwue') are missing from input DataFrame.

    Notes
    -----
    - Pools and fluxes are updated daily based on allocation rules, turnover, and disturbance detection.
    - For crops, pools are reset to zero at harvest events.

    ToDo
    -----
    - Refine crop modelling --> growing_season_limit necessary ?
    _ Add PC output for RothC when crop harvested or not emerged
    - Add grazing -> manure return to RothC
    """

    # Extract variables from DataFrame
    soil_moisture = df["soil_moisture"].values
    gpp = df["gpp"].values
    iwue = df["iwue"].values
    lue = df["lue"].values
    temp = df["temp_degC"].values
    vpd = df["vpd_Pa"].values
    doy = df.index.dayofyear.values
    lai_obs = df["lai_obs"].values  # assumed present

    # Calculate timestep in days (assume regular interval)
    if len(df.index) > 1:
        ts = (df.index[1] - df.index[0]).days
    else:
        ts = 1  # fallback to 1 day if only one row

    # --- Handle new user inputs: lue and iwue ---
    if lue is None or iwue is None:
        raise ValueError("Both lue and iwue timeseries must be provided as inputs.")

    def norm01(x):
        min_x = np.nanmin(x)
        max_x = np.nanmax(x)
        if max_x > min_x:
            return (x - min_x) / (max_x - min_x)
        else:
            return np.zeros_like(x)

    lue_norm = norm01(lue)
    iwue_norm = norm01(iwue)

    iwue_norm_inv = 1 - iwue_norm
    cue_raw = 0.5 * (lue_norm + iwue_norm_inv)
    cue = 0.2 + cue_raw * (0.9 - 0.2)

    initial_pools = {
        "tree": {"leaves": ini_pool[0], "stem": ini_pool[1], "roots": ini_pool[2]},
        "grass": {"leaves": ini_pool[0], "stem": ini_pool[1], "roots": ini_pool[2]},
        "crop": {"leaves": ini_pool[0], "stem": ini_pool[1], "roots": ini_pool[2]},
        "shrub": {"leaves": ini_pool[0], "stem": ini_pool[1], "roots": ini_pool[2]},
    }

    if plant_type.lower() not in initial_pools:
        print(
            f"Error: Unsupported plant type '{plant_type}'. Please choose 'tree', 'grass', 'crop', or 'shrub'."
        )
        return None

    pools = initial_pools[plant_type.lower()]
    leaves_pool = [pools["leaves"]]
    stem_pool = [pools["stem"]]
    roots_pool = [pools["roots"]]
    npp = []
    litter2soil = []
    roots_respiration_loss_daily = []
    stem_respiration_loss_daily = []
    leaves_respiration_loss_daily = []
    leaf_area_index = []
    disturbance_pool = []

    allocation_bases = {
        "tree": {"leaves": 0.05, "stem": 0.65, "roots": 0.30},
        "grass": {"leaves": 0.40, "stem": 0.10, "roots": 0.50},
        "shrub": {"leaves": 0.10, "stem": 0.40, "roots": 0.50},
        "crop": {"leaves": 0.25, "stem": 0.5, "roots": 0.25},
    }
    base = allocation_bases[plant_type.lower()]

    soil_moisture_drought_threshold_mm = np.percentile(soil_moisture, 25)
    vpd_max = np.percentile(vpd, 75)

    # --- Growing season estimation ---
    growing_season = temp > growing_season_limit  # boolean mask

    # --- Disturbance detection setup ---
    gpp_rel_change = np.zeros_like(gpp)
    lai_rel_change = np.zeros_like(lai_obs)
    gpp_rel_change[1:] = (gpp[1:] - gpp[:-1]) / np.maximum(gpp[:-1], 1e-6)
    lai_rel_change[1:] = (lai_obs[1:] - lai_obs[:-1]) / np.maximum(lai_obs[:-1], 1e-6)

    for i in range(len(gpp)):
        seasonality_mod = np.sin(2 * np.pi * doy[i] / 365.0)
        temp_mod = (temp[i] - 20) / 100

        dynamic_percentages = {
            "leaves": max(0, base["leaves"] + 0.15 * seasonality_mod + 0.1 * temp_mod),
            "roots": max(0, base["roots"] - 0.15 * seasonality_mod - 0.05 * temp_mod),
            "stem": max(0, base["stem"] - 0.05 * temp_mod),
        }

        total_dynamic_percentage = (
            dynamic_percentages["leaves"]
            + dynamic_percentages["stem"]
            + dynamic_percentages["roots"]
        )
        if total_dynamic_percentage > 0:
            for part in dynamic_percentages:
                dynamic_percentages[part] /= total_dynamic_percentage

        normalized_moisture = min(
            soil_moisture[i] / soil_moisture_drought_threshold_mm, 1.0
        )
        normalized_vpd = min(vpd[i] / vpd_max, 1.0)
        drought_modifier = (1 - normalized_moisture) + normalized_vpd

        root_adjustment = drought_modifier * 0.1
        leaf_stem_adjustment = -drought_modifier * 0.1

        final_percentages = {
            "roots": max(0, dynamic_percentages["roots"] + root_adjustment),
            "leaves": max(
                0, dynamic_percentages["leaves"] + leaf_stem_adjustment * 0.7
            ),
            "stem": max(0, dynamic_percentages["stem"] + leaf_stem_adjustment * 0.3),
        }

        total_percentage = (
            final_percentages["roots"]
            + final_percentages["leaves"]
            + final_percentages["stem"]
        )
        if total_percentage > 0:
            for part in final_percentages:
                final_percentages[part] /= total_percentage

        # --- CROP-SPECIFIC: No pools before GPP > 1 ---
        if (
            plant_type.lower() == "crop"
            and gpp[i] <= 1.0
            and (leaves_pool[-1] + stem_pool[-1] + roots_pool[-1]) == 0.0
        ):
            # No pools, no allocation, all outputs zero
            leaves_pool.append(0.0)
            stem_pool.append(0.0)
            roots_pool.append(0.0)
            litter2soil.append(0.0)
            leaves_respiration_loss_daily.append(0.0)
            stem_respiration_loss_daily.append(0.0)
            roots_respiration_loss_daily.append(0.0)
            leaf_area_index.append(0.0)
            npp.append(0.0)
            disturbance_pool.append(0.0)
            continue

        allocated_gpp = {
            "leaves": gpp[i] * final_percentages["leaves"] * ts,
            "stem": gpp[i] * final_percentages["stem"] * ts,
            "roots": gpp[i] * final_percentages["roots"] * ts,
        }

        leaves_respiration_loss = allocated_gpp["leaves"] * (1 - cue[i])
        stem_respiration_loss = allocated_gpp["stem"] * (1 - cue[i])
        roots_respiration_loss = allocated_gpp["roots"] * (1 - cue[i])

        litter_cue_modifier = 1 + (1 - cue[i])
        litter_carbon_from_leaves = (
            leaves_pool[-1]
            * (1 - (1 - leaf_turnover_rate) ** ts)
            / ts
            * litter_cue_modifier
        )
        litter_carbon_from_roots = (
            roots_pool[-1]
            * (1 - (1 - root_turnover_rate) ** ts)
            / ts
            * litter_cue_modifier
        )
        litter_carbon_from_stem = (
            stem_pool[-1]
            * (1 - (1 - stem_turnover_rate) ** ts)
            / ts
            * litter_cue_modifier
        )

        total_litter_carbon = (
            litter_carbon_from_leaves
            + litter_carbon_from_roots
            + litter_carbon_from_stem
        )

        # --- Disturbance/Harvest detection and application ---
        disturbance = 0.0
        if (
            i > 0
            and growing_season[i]
            and gpp_rel_change[i] < -disturbance_limit
            and lai_rel_change[i] < -disturbance_limit
        ):
            if plant_type.lower() == "crop":
                # HARVEST: Remove all leaves, add all leaves, stem, roots to litter
                disturbance = leaves_pool[-1]  # all leaves removed
                total_litter_carbon += leaves_pool[-1] + stem_pool[-1] + roots_pool[-1]
                leaves_pool[-1] = 0.0
                stem_pool[-1] = 0.0
                roots_pool[-1] = 0.0
            else:
                # Remove a fraction of leaves proportional to the mean relative decline
                frac = np.mean([abs(gpp_rel_change[i]), abs(lai_rel_change[i])])
                frac = min(frac, 1.0)
                disturbance = leaves_pool[-1] * frac
                leaves_pool[-1] -= disturbance
        disturbance_pool.append(disturbance)

        # Update pools (for crop, pools already set to zero after harvest)
        if not (
            plant_type.lower() == "crop"
            and i > 0
            and growing_season[i]
            and gpp_rel_change[i] < -disturbance_limit
            and lai_rel_change[i] < -disturbance_limit
        ):
            leaves_pool.append(
                leaves_pool[-1]
                + allocated_gpp["leaves"]
                - litter_carbon_from_leaves
                - leaves_respiration_loss
            )
            stem_pool.append(
                stem_pool[-1]
                + allocated_gpp["stem"]
                - litter_carbon_from_stem
                - stem_respiration_loss
            )
            roots_pool.append(
                roots_pool[-1]
                + allocated_gpp["roots"]
                - litter_carbon_from_roots
                - roots_respiration_loss
            )
        else:
            leaves_pool.append(leaves_pool[-1])
            stem_pool.append(stem_pool[-1])
            roots_pool.append(roots_pool[-1])

        litter2soil.append(total_litter_carbon)
        leaves_respiration_loss_daily.append(leaves_respiration_loss)
        stem_respiration_loss_daily.append(stem_respiration_loss)
        roots_respiration_loss_daily.append(roots_respiration_loss)

        lai = leaves_pool[-1] / lca
        leaf_area_index.append(lai)

        npp.append(
            gpp[i] * ts
            - leaves_respiration_loss
            - stem_respiration_loss
            - roots_respiration_loss
        )

    output = {
        "leaves": np.array(leaves_pool[1:]),
        "stem": np.array(stem_pool[1:]),
        "roots": np.array(roots_pool[1:]),
        "litter2soil": np.array(litter2soil),
        "leaves_respiration_loss": np.array(leaves_respiration_loss_daily),
        "stem_respiration_loss": np.array(stem_respiration_loss_daily),
        "roots_respiration_loss": np.array(roots_respiration_loss_daily),
        "leaf_area_index": np.array(leaf_area_index),
        "npp": np.array(npp),
        "cue": cue,
        "disturbance": np.array(disturbance_pool),
    }

    df_out = pd.DataFrame(index=df.index)
    for key, arr in output.items():
        df_out[key] = arr

    return df_out
