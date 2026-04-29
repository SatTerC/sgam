---
title: Home
icon: lucide/house
---

# SGAM: Simplified Growth and Allocation Model

SGAM is a Python implementation of a simplified plant growth and carbon allocation model.
Given weekly environmental drivers — gross primary productivity (GPP), temperature, soil moisture, vapour pressure deficit (VPD), light use efficiency (LUE), and intrinsic water use efficiency (iWUE) — it simulates the allocation of carbon to leaf, stem, and root biomass pools for four plant functional types: tree, grass, shrub, and crop.
It accounts for dynamic allocation, autotrophic respiration, litterfall turnover, and disturbance or harvest events, and enforces strict mass balance at every timestep.

SGAM is designed as a component of the [SatTerC](https://github.com/SatTerC/satterc) modular terrestrial carbon modelling framework, where it sits between a photosynthesis model (which produces GPP) and a soil carbon model (which consumes litter inputs).
It can also be used as a standalone package.

## Documentation

- [Getting Started](getting-started.md) — installation and a worked example
- [Science](science.md) — model description and equations
- [API Reference](API_Reference/sgam.md) — full class and function documentation
