#!/usr/bin/env python3
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
        "estimator": [KernelRidgeRegressor],
        "set_scaling": np.linspace(0.01, 0.1, 10),
        "lattice_resolution": [100, 200, 300, 400],
        "feature_sigma_l": np.linspace(0.01, 1.0, 25),
        "num_frequencies": [4, 5, 6, 7, 8, 9],
    }
    run_grid(parse_args("benchmarks/integration/linear.yaml"), grid=grid)

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
