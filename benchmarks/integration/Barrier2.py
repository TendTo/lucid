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
        "seed": [42],
        "num_frequencies": [6, 7, 8],
        "lattice_resolution": [330],
        "set_scaling": [0.03, 0.04, 0.05],
        "feature_sigma_l": [
            np.array([sigma_l1, sigma_l2])
            for sigma_l1, sigma_l2 in itertools.product(np.linspace(0.04, 0.4, 10), np.linspace(0.0005, 0.15, 10))
        ],
    }
    run_grid(parse_args("benchmarks/integration/barrier2.yaml"), grid=grid)

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
