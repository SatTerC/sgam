---
title: Getting Started
icon: lucide/rocket
---

# Getting Started

## Installation

Install the package via `pip` or `uv`. Currently it is only available from GitHub.

=== "pip"

    ``` sh
    pip install git+https://github.com/satterc/sgam
    ```

=== "uv"

    ``` sh
    uv add git+https://github.com/satterc/sgam
    ```

## Basic usage

The main entry point is the `Sgam` class. Instantiate it with a plant functional
type, then call `forward` (or the instance directly) with weekly time-series arrays
of environmental drivers and initial pool sizes.

```python
import numpy as np
from sgam import Sgam
from sgam.pft import PlantFunctionalType

# Initialise for a temperate deciduous tree
model = Sgam(plant_type=PlantFunctionalType.TREE)

# Simulate 52 weeks with constant driving data
n_weeks = 52
output = model.forward(
    gpp=np.full(n_weeks, 8.0),           # gC week⁻¹
    temperature=np.full(n_weeks, 15.0),  # °C
    soil_moisture=np.full(n_weeks, 0.3), # m³ m⁻³
    vpd=np.full(n_weeks, 800.0),         # Pa
    lue=np.full(n_weeks, 0.5),           # gC MJ⁻¹
    iwue=np.full(n_weeks, 80.0),         # µmol mol⁻¹
    week_of_year=np.arange(1, n_weeks + 1, dtype=float),
    disturbances=np.zeros(n_weeks),      # no disturbance events
    leaf_pool_init=50.0,                 # gC
    stem_pool_init=200.0,                # gC
    root_pool_init=100.0,                # gC
)

# Pool sizes and fluxes are available as named attributes
print(output.pools.leaf[-1])   # final leaf pool size (gC)
print(output.npp.stem.sum())   # cumulative stem NPP (gC)
```

The returned [`SgamOutput`](API_Reference/sgam.md#sgam.sgam.SgamOutput) object contains:

| Attribute | Description |
|---|---|
| `pools` | Time series of leaf, stem, root, litter, and removed-carbon pool sizes |
| `npp` | Net primary productivity flux to each pool |
| `turnover` | Litterfall flux from each pool |
| `respiration` | Autotrophic respiration flux from each pool |
| `disturbance` | Carbon removed by disturbance or harvest events |
| `diagnostics` | Derived quantities: CUE, allocation fractions, drought modifier |

## Disturbance detection

The [`Disturbances`](API_Reference/disturbance.md#sgam.disturbance.Disturbances) class can identify disturbance
events (harvest, fire, pest damage) from rapid declines in daily GPP and LAI, and
convert them into the weekly severity array expected by `Sgam.forward`.

```python
from sgam import Disturbances

detector = Disturbances(growing_season_limit=5.0, disturbance_threshold=0.3)
weekly_severity = detector.forward(
    temperature=daily_temperature,
    gpp=daily_gpp,
    lai=daily_lai,
    aggregate=True,  # aggregate daily output to weekly timesteps
)
```

## Plant functional types and parameters

Four plant functional types are available via the `PlantFunctionalType` enum:
`TREE`, `GRASS`, `SHRUB`, and `CROP`. Each has a set of default physiological
parameters (`PftParams`) covering allocation fractions, turnover rates, drought
sensitivities, and light/water use efficiencies. Custom parameters can be passed
directly to `Sgam`:

```python
import dataclasses
from sgam.pft import PftParams, get_default_pft_params

# Start from the defaults and override individual fields
base = get_default_pft_params(PlantFunctionalType.GRASS)
custom_params = dataclasses.replace(base, leaf_turnover_rate=0.05)

model = Sgam(plant_type=PlantFunctionalType.GRASS, pft_params=custom_params)
```

To build an entirely bespoke parameter set, use `dataclasses.asdict` on a default
instance to get a complete dict of all parameters, merge in your overrides using
the `|` operator, then pass the result to the `PftParams` constructor:

```python
import dataclasses
from sgam.pft import PftParams, PlantFunctionalType, get_default_pft_params

base_dict = dataclasses.asdict(get_default_pft_params(PlantFunctionalType.TREE))

custom_dict = {
    "wilting_point": 0.12,       # more drought-tolerant than the default
    "vpd_threshold": 1500.0,     # Pa — stomata stay open longer under dry air
    "disturbance_leaf_loss_frac": 0.6,  # higher leaf loss per disturbance event
}

custom_params = PftParams(**(base_dict | custom_dict))
model = Sgam(plant_type=PlantFunctionalType.TREE, pft_params=custom_params)
```

See the [API Reference](API_Reference/pft.md) for all available parameters.
