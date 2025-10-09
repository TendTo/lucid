#!/usr/bin/env python3
import argparse
import time

import numpy as np

from pylucid import *
from pylucid import __version__


class Args:
    config: str
    confidence_level: float = 0.9
    num_samples: int = 1000
    verbosity: int = log.LOG_WARN


def scenario_config(args: Args):
    # ################################## #
    # System dynamics
    # ################################## #
    config = Configuration.from_file(f"benchmarks/integration/{args.config}")

    log.set_verbosity(args.verbosity)
    if config.seed >= 0:
        np.random.seed(config.seed)  # For reproducibility
        random.seed(config.seed)

    def f(x: np.ndarray) -> np.ndarray:
        # Ensure x is 2D array for consistent noise addition
        if x.ndim == 1:
            x = x[:, np.newaxis]
        return config.system_dynamics(x) + (np.random.normal(scale=config.noise_scale, size=x.shape))

    bounds = MontecarloSimulation().safety_probability(
        X_bounds=config.X_bounds,
        X_init=config.X_init,
        X_unsafe=config.X_unsafe,
        system_dynamics=f,
        time_horizon=config.time_horizon,
        confidence_level=args.confidence_level,
        num_samples=args.num_samples,
    )

    log.info(f"Safety probability bounds: {bounds}")


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Run MonteCarlo baseline benchmark.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    parser.add_argument(
        "-c",
        "--confidence_level",
        type=float,
        default=0.9,
        help="Confidence level for the safety probability estimation.",
    )
    parser.add_argument("-n", "--num_samples", type=int, default=1000, help="Number of Monte Carlo samples to explore.")
    parser.add_argument("-v", "--verbosity", type=int, default=log.LOG_WARN, help="Logging verbosity level.")
    return parser.parse_args()


if __name__ == "__main__":
    # ################################## #
    # Lucid
    # ################################## #
    log.info(f"Running baseline (LUCID version: {__version__})")
    start = time.time()

    args = parse_args()
    scenario_config(args)

    end = time.time()
    log.info(f"Elapsed time: {end - start}")
