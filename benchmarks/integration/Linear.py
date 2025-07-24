#!/usr/bin/env python3
import itertools
import multiprocessing
import time

import numpy as np
from benchmark import single_benchmark

from pylucid import *
from pylucid import __version__


def scenario_config(param_name: tuple[str], param_combinations: tuple[tuple]) -> Configuration:
    """Benchmark scenario taken from
    https://github.com/oxford-oxcav/fossil/blob/10f1f071784d16b2a5ee5da2f51ff2a81d753e2e/experiments/benchmarks/models.py#L350C1-L360C1
    """
    # ################################## #
    # System dynamics
    # ################################## #
    # Algorithm parameters
    config = Configuration(
        seed=42,
        gamma=15.0,
        time_horizon=15,
        num_samples=1000,
        lambda_=1e-6,
        sigma_f=1.0,
        sigma_l=np.array([1.75555556]),
        num_frequencies=5,
        plot=True,
        verify=True,
        iis_log_file="iis.ilp",
        oversample_factor=64.0,
        c_coefficient=1.0,
        system_dynamics=lambda x: x / 2.0,  # Example system dynamics
        X_bounds=RectSet([[-1, 1]]),  # State space X
        # Initial set X_0
        X_init=RectSet([[-0.5, 0.5]]),
        # Unsafe set X_U
        X_unsafe=MultiSet(RectSet([[-1, -0.9]]), RectSet([[0.9, 1]])),
        noise_scale=0.01,
        optimiser=GurobiOptimiser,
    )

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

    for key, value in zip(param_name, param_combinations):
        setattr(config, key, value)
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
        "c_coefficient": [0.2, 1.0],
        "time_horizon": [5, 10],
        "oversample_factor": [20.0, 40.0, 60.0],
    }

    param_combinations = list(itertools.product(*grid.values()))
    grid_keys = list(grid.keys())

    # Prepare arguments for multiprocessing
    args_list = [(grid_keys, param_combination) for param_combination in param_combinations]

    # Run benchmarks in parallel using multiprocessing
    with multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 2)) as pool:
        pool.starmap(scenario_config, args_list)

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
