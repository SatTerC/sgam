---
title: Home
icon: lucide/house
---

# SGAM: Simplified Growth/GPP Allocation Model


## Installation

Install the package via `pip` or `uv`.
Currently it is only available from GitHub.

=== "pip"

    ``` sh
    pip install git+https://github.com/satterc/sgam
    ```

=== "uv"

    ``` sh
    uv add git+https://github.com/satterc/sgam
    ```

This will install a package called `sgam` into your environment.

## Basic usage

`sgam` provides a class, `SgamComponent`, which provides the main user interface to the model implementation.

```python
from sgam import SgamComponent

# To do
params = ...
forward_data = ...

model = SgamComponent(**params)

```

For more details on the contents of this package, see the API reference.
