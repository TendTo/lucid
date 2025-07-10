#!/usr/bin/env python3
import time

import numpy as np

from pylucid import *
from pylucid import __version__
from pylucid.pipeline import pipeline


def scenario_config(
    args: Configuration = Configuration(
        seed=42,
        gamma=1,
        time_horizon=5,
        num_samples=1000,
        lambda_=1e-3,
        sigma_f=15.0,
        sigma_l=np.array([1.75555556]),
        num_frequencies=4,
        plot=True,
        verify=True,
        problem_log_file="problem.lp",
        iis_log_file="iis.ilp",
        oversample_factor=32.0,
    )
) -> "ScenarioConfig":
    # ################################## #
    # System dynamics                    #
    # ################################## #

    f_det = lambda x: 1 / 2 * x
    # Add process noise
    f = lambda x: f_det(x) + np.random.normal(scale=0.4)

    # ################################## #
    # Safety specification               #
    # ################################## #

    X_bounds = RectSet([(-1, 1)])  # State space
    X_init = RectSet([(-0.5, 0.5)])  # Initial set
    X_unsafe = MultiSet(  # Unsafe set
        RectSet([(-1, -0.9)]),
        RectSet([(0.9, 1)]),
    )

    # ################################## #
    # Sampling                           #
    # ################################## #

    x_samples = X_bounds.sample(args.num_samples)
    xp_samples = f(x_samples)

    # ################################## #
    # Lucid
    # ################################## #

    # De-comment the tuner you want to use or leave it empty to avoid tuning.
    estimator = KernelRidgeRegressor(
        kernel=GaussianKernel(sigma_f=args.sigma_f, sigma_l=args.sigma_l),
        regularization_constant=args.lambda_,
    )
    feature_map = LinearTruncatedFourierFeatureMap(
        num_frequencies=args.num_frequencies,
        sigma_l=args.sigma_l,
        sigma_f=args.sigma_f,
        x_limits=X_bounds,
    )
    # estimator = ModelEstimator(f=lambda x: feature_map(f_det(x)))  # Use the custom model estimator

    return ScenarioConfig(
        x_samples=x_samples,
        xp_samples=xp_samples,
        X_bounds=X_bounds,
        X_init=X_init,
        X_unsafe=X_unsafe,
        T=args.time_horizon,
        gamma=args.gamma,
        feature_map=feature_map,  # The feature map used to transform the input data
        f_det=f_det,  # The deterministic part of the system dynamics
        estimator=estimator,  # The estimator used to model the system dynamics
        sigma_f=args.sigma_f,  # Signal variance parameter for the kernel
        problem_log_file=args.problem_log_file,  # The lp file containing the optimization problem
        iis_log_file=args.iis_log_file,  # The ilp file containing the irreducible infeasible set (IIS) if the problem is infeasible
        plot=args.plot,  # Whether to plot the results
        verify=args.verify,  # Whether to verify the barrier certificate using dReal
        oversample_factor=args.oversample_factor,  # Factor by which to oversample the frequency space
    )


if __name__ == "__main__":
    log.info(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()
    pipeline(**scenario_config())
    end = time.time()
    log.info(f"Elapsed time: {end - start}")
