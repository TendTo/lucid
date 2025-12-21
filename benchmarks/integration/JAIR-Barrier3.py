#!/usr/bin/env python3
import numpy as np
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
    filename = "benchmarks/integration/jair-barrier3.yaml"

    grid = {
        "num_samples": [1000],
        "seed": [42],
        "num_frequencies": [5, 7, 9, 11, 13],
        "oversample_factor": [10, 20, 30],
        "set_scaling": [0.02],
        "feature_sigma_l": [np.array([0.05, 0.15])],
        # "feature_sigma_l": [
        #     np.array([sigma_l1, sigma_l2])
        #     for sigma_l1, sigma_l2 in itertools.product(np.linspace(0.05, 1.0, 20), repeat=2)
        # ],
    }
    run_grid(parse_args(filename), grid=grid)

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
