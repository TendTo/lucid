#!/usr/bin/env python3
from pylucid import *
from pylucid import __version__
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pylucid._pylucid import ParameterValuesType

def scenario_tuner(file_path: str, grid: "dict[Parameter, ParameterValuesType]"):
    action = ConfigAction(option_strings=None, dest="input")
    config = Configuration()
    action(None, config, Path(file_path), None)
    tuner = GridSearchTuner(parameters=grid)
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

    if config.system_dynamics:
        new_x_samples = config.X_bounds.sample(1000)
        new_xp_samples = config.system_dynamics(new_x_samples) + np.random.normal(scale=config.noise_scale)

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
    log.info(f"Estimator {estimator} score: {s}")
    if config.system_dynamics:
        s2 = estimator.score(new_x_samples, feature_map(new_xp_samples))
        log.info(f"Estimator {estimator} score: {s} and {s2}")
