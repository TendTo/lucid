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
        "num_frequencies": [11, 13],
        "lattice_resolution": [400, 500, 600, 700],
    }
    param_combinations = list(itertools.product(*grid.values()))
    grid_keys = list(grid.keys())

    seed = [42]
    pairs = [
        (16, 300),
        (16, 400),
        (17, 300),
        (17, 400),
        (15, 300),
        (15, 500),
        (15, 600),
        (15, 700),
        (20, 400),
        (20, 500),
        (20, 600),
        (11, 800),
        (13, 800),
        # (17, 800),
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
    for args in args_list:
        print(f"Running scenario with: {args}")
        scenario_config("benchmarks/integration/barrier2.yaml", *args)

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
