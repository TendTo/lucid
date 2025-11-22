#!/usr/bin/env python3
import itertools
import multiprocessing
import time

from benchmark import scenario_config

from pylucid import *
from pylucid import __version__

if __name__ == "__main__":
    # ################################## #
    # Lucid
    # ################################## #
    log.info(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()
    config = Configuration.from_file("benchmarks/integration/barrier3.yaml")

    grid = {
        "num_samples": [10],
        "seed": [42],
        "num_frequencies": [6, 7, 8],
        "lattice_resolution": [330],
        "set_scaling": [0.015, 0.02],
        "feature_sigma_l": [np.array([0.05, 0.1]), np.array([0.06, 0.1]), np.array([0.05, 0.15])],
    }

    param_combinations = list(itertools.product(*grid.values()))
    grid_keys = list(grid.keys())

    print(f"Running {len(param_combinations)} configurations.")
    print(f"{(param_combinations)}")

    # Prepare arguments for multiprocessing
    args_list = [(grid_keys, param_combination) for param_combination in param_combinations]

    # Run benchmarks in parallel using multiprocessing
    # MAX_PARALLEL = multiprocessing.cpu_count() // 3
    # MAX_PARALLEL = 1
    # with multiprocessing.Pool(processes=max(1, MAX_PARALLEL)) as pool:
    #     pool.starmap(scenario_config, [("benchmarks/integration/barrier3.yaml",) + args for args in args_list])
    for args in args_list:
        print(f"Running scenario with: {args}")
        scenario_config("benchmarks/integration/barrier3.yaml", *args)

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
