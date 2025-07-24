#!/usr/bin/env python3
import time

import numpy as np
from benchmark import benchmark

from pylucid import *
from pylucid import __version__


def scenario_config():
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
            RectSet((-1.8, -0.1), (-1.2, 0.1)),
            RectSet((-1.4, -0.5), (-1.2, 0.1)),
        ),
        # Unsafe set X_U
        X_unsafe=MultiSet(RectSet((-2.9, 0.1), (-2.8, 0.5)), RectSet((-2.9, 0.1), (-2.7, 0.3))),
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

    benchmark(
        name="Barrier3",
        config=config,
        grid={
            "c_coefficient": [0.2, 1.0],
            "time_horizon": [5, 10],
            "oversample_factor": [20.0, 40.0],
        },
    )

    return config


if __name__ == "__main__":
    # ################################## #
    # Lucid
    # ################################## #
    log.info(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()
    scenario_config()
    end = time.time()
    log.info(f"Elapsed time: {end - start}")
