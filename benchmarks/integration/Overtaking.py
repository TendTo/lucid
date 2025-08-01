#!/usr/bin/env python3
import itertools
import multiprocessing
import time

from benchmark import single_benchmark

from pylucid import *
from pylucid import __version__


def scenario_config(param_name: tuple[str], param_combinations: tuple[tuple]) -> Configuration:
    # ################################## #
    # System dynamics
    # ################################## #
    action = ConfigAction(option_strings=None, dest="input")
    config = Configuration()
    action(None, config, Path("benchmarks/integration/overtaking.yaml"), None)

    for key, value in zip(param_name, param_combinations):
        setattr(config, key, value)

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
        "num_frequencies": [5, 6],
        "num_oversample": [100, 110],
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
