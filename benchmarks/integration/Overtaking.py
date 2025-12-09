#!/usr/bin/env python3
import itertools
import time

from benchmark import parse_args, run_grid

from pylucid import *
from pylucid import __version__

if __name__ == "__main__":
    # ################################## #
    # Lucid
    # ################################## #
    log.info(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()

    grid = {
        "num_samples": [1000],
        "seed": [42],
        "num_frequencies": [5, 6, 7],
        "lattice_resolution": [70],
        "set_scaling": [0.04, 0.05],
        "feature_sigma_l": [
            np.array([sigma_l1, sigma_l2, sigma_l3])
            for sigma_l1, sigma_l2, sigma_l3 in itertools.product(np.linspace(0.05, 1.0, 3), repeat=3)
        ],
    }
    run_grid(parse_args("benchmarks/integration/overtaking.yaml"), grid=grid)

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
