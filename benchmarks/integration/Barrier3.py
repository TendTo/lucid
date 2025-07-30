#!/usr/bin/env python3
import itertools
import multiprocessing
import time

import numpy as np
from benchmark import grid_to_config, single_benchmark

from pylucid import *
from pylucid import __version__
from pylucid.plot import plot_function, plot_data


def scenario_config(param_name: tuple[str], param_combinations: tuple[tuple]) -> Configuration:
    """Benchmark scenario taken from
    https://github.com/oxford-oxcav/fossil/blob/10f1f071784d16b2a5ee5da2f51ff2a81d753e2e/experiments/benchmarks/models.py#L350C1-L360C1
    """
    # ################################## #
    # System dynamics
    # ################################## #
    action = ConfigAction(option_strings=None, dest="input")
    config = Configuration()
    action(None, config, Path("benchmarks/integration/barrier3.yaml"), None)

    for key, value in zip(param_name, param_combinations):
        setattr(config, key, value)

    # Add process noise
    if config.seed >= 0:
        np.random.seed(config.seed)  # For reproducibility
        random.seed(config.seed)

    # ################################## #
    # Data
    # ################################## #
    f = lambda x: config.system_dynamics(x) + (np.random.normal(scale=config.noise_scale))
    config.x_samples = config.X_bounds.sample(config.num_samples)
    config.xp_samples = f(config.x_samples)

    # feature_map = config.feature_map(
    #     num_frequencies=config.num_frequencies,
    #     sigma_l=config.sigma_l,
    #     sigma_f=config.sigma_f,
    #     x_limits=config.X_bounds,
    # )
    # config.estimator = ModelEstimator(
    #     lambda x: feature_map(f(x)),
    #     {
    #         Parameter.SIGMA_F: config.sigma_f,
    #         Parameter.SIGMA_L: config.sigma_l,
    #         Parameter.REGULARIZATION_CONSTANT: config.lambda_,
    #     },
    # )

    # ################################## #
    # Running the pipeline
    # ################################## #

    if config.num_frequencies >= 12 and config.oversample_factor > 20.0:
        log.warn("The number of frequencies and oversampling factor too high.")
        return

    single_benchmark(
        name="Barrier3",
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
        "num_frequencies": [18],
        # "num_frequencies": [4],
        "c_coefficient": [1.0],
        "time_horizon": [5],
        # "oversample_factor": [1.0],
        "num_oversample": [350],
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
