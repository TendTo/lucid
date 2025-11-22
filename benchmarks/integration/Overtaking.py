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
        "seed": [42],
        "num_frequencies": [3, 4],
        "lattice_resolution": [100, 110],
    }

    param_combinations = list(itertools.product(*grid.values()))
    grid_keys = list(grid.keys())

    seed = [42]
    pairs = [
        # (5, 99),
        # (5, 100),
        # (5, 88),
        # (5, 77),
        (5, 66),
        (4, 90),
        (4, 81),
        (5, 55),
        (4, 72),
        (5, 44),
        (4, 63),
    ]

    param_combinations = [(s, f, o) for s, (f, o) in itertools.product(seed, pairs)]
    print(f"Running {len(param_combinations)} configurations.")
    print(f"{(param_combinations)}")

    # Prepare arguments for multiprocessing
    args_list = [(grid_keys, param_combination) for param_combination in param_combinations]

    # Run benchmarks in parallel using multiprocessing
    MAX_PARALLEL = multiprocessing.cpu_count() // 3
    MAX_PARALLEL = 1
    with multiprocessing.Pool(processes=max(1, MAX_PARALLEL)) as pool:
        pool.starmap(scenario_config, args_list)
    # scenario_config(*args_list[0])  # Run only one configuration for testing

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
