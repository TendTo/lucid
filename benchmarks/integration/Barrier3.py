#!/usr/bin/env python3
import itertools
import multiprocessing
import time

import numpy as np
from benchmark import grid_to_config, single_benchmark

from pylucid import *
from pylucid import __version__
from pylucid.plot import plot_function


def scenario_config(param_name: tuple[str], param_combinations: tuple[tuple]) -> Configuration:
    """Benchmark scenario taken from
    https://github.com/oxford-oxcav/fossil/blob/10f1f071784d16b2a5ee5da2f51ff2a81d753e2e/experiments/benchmarks/models.py#L350C1-L360C1
    """
    # ################################## #
    # System dynamics
    # ################################## #
    config = Configuration(
        seed=42,
        gamma=2.0,
        time_horizon=5,
        num_samples=1000,
        lambda_=1e-6,
        sigma_f=15.0,
        sigma_l=np.array([1.0, 1.0]),
        num_frequencies=4,
        plot=True,
        verify=True,
        iis_log_file="iis.ilp",
        oversample_factor=40.0,
        c_coefficient=1.0,
        system_dynamics=lambda x: np.array([x[:, 1], -x[:, 0] - x[:, 1] + 1 / 3 * x[:, 0] ** 3]).T,  # lambda x: x
        X_bounds=RectSet((-3, -2), (2.5, 1)),  # State space X
        # Initial set X_0
        X_init=MultiSet(
            RectSet((1, -0.5), (2, 0.5)),
            RectSet((-1.8, -0.1), (-1.4, 0.1)),
            RectSet((-1.4, -0.5), (-1.2, 0.1)),
        ),
        # Unsafe set X_U
        # X_unsafe=MultiSet(RectSet((0.4, 0.1), (0.6, 0.5)), RectSet((0.6, 0.1), (0.8, 0.3))),
        # X_unsafe=MultiSet(RectSet((0.4, 0.05), (0.6, 0.45)), RectSet((0.6, 0.05), (0.7, 0.3))),
        X_unsafe=MultiSet(RectSet((0.4, 0.2), (0.6, 0.6)), RectSet((0.6, 0.2), (0.7, 0.4))),
    )

    for key, value in zip(param_name, param_combinations):
        setattr(config, key, value)

    # plot_function(
    #     f=config.system_dynamics,
    #     X_bounds=config.X_bounds,
    #     X_init=config.X_init,
    #     X_unsafe=config.X_unsafe,
    # )

    # Add process noise
    if config.seed >= 0:
        np.random.seed(config.seed)  # For reproducibility
        random.seed(config.seed)
    f = lambda x: config.system_dynamics(x) # + (np.random.normal(scale=config.noise_scale))
    config.estimator = ModelEstimator(f)

    # ################################## #
    # Data
    # ################################## #
    config.x_samples = config.X_bounds.sample(config.num_samples)
    config.xp_samples = f(config.x_samples)

    # ################################## #
    # Running the pipeline
    # ################################## #

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
        "num_frequencies": [4, 5, 8, 9, 12, 13, 16, 17],
        "c_coefficient": [0.2],
        "time_horizon": [5],
        "oversample_factor": [10.0, 20.0, 30.0],
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
