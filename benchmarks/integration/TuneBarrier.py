#!/usr/bin/env python3
import time

import numpy as np

from pylucid import *
from pylucid import __version__
from pylucid.pipeline import pipeline

try:
    from pylucid.plot import plot_function
except ImportError:

    def plot_function(*args, **kwargs):
        pass


"""
KernelRidgeRegressor( kernel( GaussianKernel( sigma_l( 10  7  5 ) sigma_f( 7 ) isotropic( 0 ) ) ) regularization_constant( 1e-05 ) )                 => 0.999265720310824
KernelRidgeRegressor( kernel( GaussianKernel( sigma_l( 3.57182 3.12249 1.10899 ) sigma_f( 7 ) isotropic( 0 ) ) ) regularization_constant( 1e-05 ) )  => 0.9999970679186748
KernelRidgeRegressor( kernel( GaussianKernel( sigma_l( 3.59381 0.278256 3.59381 ) sigma_f( 7 ) isotropic( 0 ) ) ) regularization_constant( 1e-05 ) ) => 0.9999999635706251
"""
import itertools

def scenario_config(config: Configuration):
    # Add process noise
    if config.seed >= 0:
        np.random.seed(config.seed)  # For reproducibility
        random.seed(config.seed)

    if config.system_dynamics:
        f = lambda x: config.system_dynamics(x) + (np.random.normal(scale=config.noise_scale))
        config.x_samples = config.X_bounds.sample(config.num_samples)
        config.xp_samples = f(config.x_samples)

    val = [v for v in np.logspace(-3, 2, num=10, endpoint=True, dtype=float)]
    tuner = GridSearchTuner(
        parameters={
            Parameter.REGULARIZATION_CONSTANT: [1e-8, 1e-9, 1e-10, 1e-11],
            Parameter.SIGMA_F: [1.0],
            Parameter.SIGMA_L: [np.array(vs) for vs in itertools.product(val, val)],
        },
    )
    estimator = KernelRidgeRegressor(
        kernel=GaussianKernel(sigma_l=config.sigma_l, sigma_f=config.sigma_f),
        regularization_constant=config.lambda_,
    )
    feature_map = LinearTruncatedFourierFeatureMap(
        num_frequencies=config.num_frequencies,
        sigma_l=config.sigma_l,
        sigma_f=config.sigma_f,
        x_limits=config.X_bounds,
    )

    estimator.fit(config.x_samples, feature_map(config.xp_samples))
    log.info(
        f"Using estimator: {estimator} with score {estimator.score(config.x_samples, feature_map(config.xp_samples))}"
    )
    log.set_verbosity(log.LOG_WARN)
    tuner.tune(
        estimator,
        config.x_samples,
        config.xp_samples,
        LinearTruncatedFourierFeatureMap,
        config.num_frequencies,
        config.X_bounds,
    )
    feature_map = LinearTruncatedFourierFeatureMap(
        num_frequencies=config.num_frequencies,
        sigma_l=estimator.get(Parameter.SIGMA_L),
        sigma_f=estimator.get(Parameter.SIGMA_F),
        x_limits=config.X_bounds,
    )
    log.set_verbosity(log.LOG_INFO)
    s = estimator.score(config.x_samples, feature_map(config.xp_samples))
    new_x_samples = config.X_bounds.sample(1000)
    new_xp_samples = f(new_x_samples) + np.random.normal(scale=config.noise_scale)
    s2 = estimator.score(config.X_bounds.sample(1000), feature_map(new_xp_samples))
    log.info(f"Estimator {estimator} score: {s} and {s2}")


if __name__ == "__main__":
    action = ConfigAction(option_strings=None, dest="input")
    config = Configuration()
    action(None, config, Path("benchmarks/integration/barrier3.yaml"), None)

    scenario_config(config)
