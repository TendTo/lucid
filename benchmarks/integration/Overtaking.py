#!/usr/bin/env python3
import itertools
import multiprocessing
import time

import numpy as np
from benchmark import grid_to_config, single_benchmark

from pylucid import *
from pylucid import __version__
from pylucid.plot import plot_function, plot_data
from pylucid.cli import ConfigAction


def scenario_config(param_name: tuple[str], param_combinations: tuple[tuple]) -> Configuration:
    # ################################## #
    # System dynamics
    # ################################## #
    action = ConfigAction(option_strings=None, dest="input")
    config = Configuration()
    action(None, config, Path("benchmarks/integration/overtaking.yaml"), None)

    for key, value in zip(param_name, param_combinations):
        setattr(config, key, value)

    # plot_data(
    #     config.x_samples,
    #     config.xp_samples,
    #     X_bounds=config.X_bounds,
    #     X_init=config.X_init,
    #     X_unsafe=config.X_unsafe,
    # )

    # ################################## #
    # Running the pipeline
    # ################################## #

    single_benchmark(
        name="Overtaking",
        config=config,
    )


if __name__ == "__main__":
    # ################################## #
    # Lucid
    # ################################## #
    log.info(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()

    grid = {
        # "num_frequencies": [9, 12, 13, 16, 17],
        "num_frequencies": [5], # [6, 7], 
        # "num_frequencies": [4],
        "c_coefficient": [1.0],
        "time_horizon": [5, 15],
        "oversample_factor": [8.0],  #  , 20.0 , 30.0
        # "oversample_factor": [10.0],
    }

    param_combinations = list(itertools.product(*grid.values()))
    grid_keys = list(grid.keys())

    # Prepare arguments for multiprocessing
    args_list = [(grid_keys, param_combination) for param_combination in param_combinations]

    # Run benchmarks in parallel using multiprocessing
    MAX_PARALLEL = multiprocessing.cpu_count() // 2
    MAX_PARALLEL = 1
    with multiprocessing.Pool(processes=max(1, MAX_PARALLEL)) as pool:
        pool.starmap(scenario_config, args_list)

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
