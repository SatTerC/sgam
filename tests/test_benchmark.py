"""Performance benchmark for Sgam.forward.

Run with: pytest tests/test_benchmark.py -v --tb=short

This is intentionally a plain pytest test (not pytest-benchmark) so it
requires no extra dependency. It fails only if runtime regresses past a
hard ceiling, giving CI a canary for accidental Python-loop additions.
"""

import time

import numpy as np

from sgam.pft import PlantFunctionalType
from sgam.sgam import Sgam

N_WEEKS = 520  # 10 years
MAX_SECONDS = 2.0  # hard ceiling — adjust after first baseline run


def make_inputs(n: int) -> dict:
    rng = np.random.default_rng(0)
    return dict(
        gpp=rng.uniform(0.0, 10.0, n),
        temperature=rng.uniform(5.0, 30.0, n),
        soil_moisture=rng.uniform(0.1, 0.5, n),
        vpd=rng.uniform(200.0, 1500.0, n),
        lue=rng.uniform(0.1, 1.0, n),
        iwue=rng.uniform(50.0, 400.0, n),
        week_of_year=np.tile(np.arange(1, 53, dtype=float), n // 52 + 1)[:n],
        disturbances=np.zeros(n),
        leaf_pool_init=1.0,
        stem_pool_init=2.0,
        root_pool_init=1.0,
    )


class TestBenchmark:
    def test_forward_520_weeks_all_pfts(self):
        inputs = make_inputs(N_WEEKS)
        t0 = time.perf_counter()
        for pft in PlantFunctionalType:
            Sgam(pft).forward(**inputs)
        elapsed = time.perf_counter() - t0
        assert elapsed < MAX_SECONDS, (
            f"forward() over {N_WEEKS} weeks × 4 PFTs took {elapsed:.2f}s "
            f"(ceiling {MAX_SECONDS}s). Check for accidental Python-loop regressions."
        )
