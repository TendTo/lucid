#!/usr/bin/env python3
import itertools
import multiprocessing
import time

import numpy as np
from benchmark import single_benchmark

from pylucid import *
from pylucid import __version__


def scenario_config(param_name: tuple[str], param_combinations: tuple[tuple]) -> Configuration:
    # ################################## #
    # System dynamics
    # ################################## #
    config = Configuration.from_file("benchmarks/integration/linear.yaml")
    for key, value in zip(param_name, param_combinations):
        setattr(config, key, value)

    # Add process noise
    if config.seed >= 0:
        np.random.seed(config.seed)  # For reproducibility
        random.seed(config.seed)
    f = lambda x: config.system_dynamics(x) + (np.random.normal(scale=config.noise_scale))

    # ################################## #
    # Data
    # ################################## #
    config.x_samples = config.X_bounds.sample(config.num_samples)
    config.xp_samples = f(config.x_samples)

    # ################################## #
    # Running the pipeline
    # ################################## #

    single_benchmark(
        name="Linear",
        config=config,
    )


if __name__ == "__main__":
    # ################################## #
    # Lucid
    # ################################## #
    log.info(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()

    grid = {
        "lattice_resolution": [704],
    }

    param_combinations = list(itertools.product(*grid.values()))
    grid_keys = list(grid.keys())

    # Prepare arguments for multiprocessing
    args_list = [(grid_keys, param_combination) for param_combination in param_combinations]

    # Run benchmarks in parallel using multiprocessing
    MAX_PARALLEL = multiprocessing.cpu_count() // 3
    with multiprocessing.Pool(processes=max(1, MAX_PARALLEL)) as pool:
        pool.starmap(scenario_config, args_list)
    # scenario_config(*args_list[0])  # Run only one configuration for testing

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
