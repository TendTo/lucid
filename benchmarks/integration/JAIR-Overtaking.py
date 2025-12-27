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
        "num_frequencies": [5],
        "oversample_factor": [8, 9, 10],
        "set_scaling": [0.02],
        "epsilon": [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001],
        # "feature_sigma_l": [
        #     np.array([sigma_l1, sigma_l2, sigma_l3])
        #     for sigma_l1, sigma_l2, sigma_l3 in itertools.product(np.linspace(0.05, 1.0, 3), repeat=3)
        # ],
    }
    run_grid(parse_args("benchmarks/integration/jair-overtaking.yaml"), grid=grid)

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
