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

    grid = {
        "num_frequencies": [7, 8, 9],
    }

    param_combinations = list(itertools.product(*grid.values()))
    grid_keys = list(grid.keys())

    # Prepare arguments for multiprocessing
    args_list = [(grid_keys, param_combination) for param_combination in param_combinations]

    # Run benchmarks in parallel using multiprocessing
    MAX_PARALLEL = multiprocessing.cpu_count() // 3
    # MAX_PARALLEL = 1
    # with multiprocessing.Pool(processes=max(1, MAX_PARALLEL)) as pool:
    #     pool.starmap(scenario_config, [("benchmarks/integration/linear.yaml",) + args for args in args_list])
    for args in args_list:
        print(f"Running scenario with parameters: {args}")
        scenario_config("benchmarks/integration/linear.yaml", *args)
    # scenario_config("benchmarks/integration/linear.yaml", *args_list[0])  # Run only one configuration for testing

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
