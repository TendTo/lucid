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
        "estimator": [KernelRidgeRegressor, ModelEstimator],
        "lattice_resolution": [100, 200],
        "num_frequencies": [7, 8, 9],
    }
    run_grid(parse_args("benchmarks/integration/linear.yaml"), grid=grid)

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
