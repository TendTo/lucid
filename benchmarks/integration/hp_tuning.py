#!/usr/bin/env python3
from typing import TYPE_CHECKING

from pylucid import *
from pylucid import __version__

if TYPE_CHECKING:
    from pylucid._pylucid import ParameterValuesType


def score_estimator(config: Configuration):
    # SCORE
    feature_map = LinearTruncatedFourierFeatureMap(
        num_frequencies=config.num_frequencies,
        sigma_l=config.estimator.get(Parameter.SIGMA_L),
        sigma_f=config.estimator.get(Parameter.SIGMA_F),
        x_limits=config.X_bounds,
    )
    config.estimator.consolidate(config.x_samples, feature_map(config.xp_samples))
    log.info(f"Estimator {config.estimator}")
    log.info(f"Score T {config.estimator.score(config.x_samples, feature_map(config.xp_samples))}")
    if config.system_dynamics:
        new_x_samples = config.X_bounds.sample(config.num_samples)
        new_xp_samples = config.system_dynamics(new_x_samples) + (np.random.normal(scale=config.noise_scale))
        log.info(f"Score new {config.estimator.score(new_x_samples, feature_map(new_xp_samples))}")


def lbfgs_tuner(config: Configuration):
    tuner = LbfgsTuner(
        lb=[1e-5] * config.X_bounds.dimension,
        ub=[1e5] * config.X_bounds.dimension,
        parameters=LbfgsParameters(min_step=0, linesearch=5),
    )
    config.estimator.fit(config.x_samples, config.xp_samples, tuner=tuner)
    score_estimator(config)


def grid_search_tuner(config: Configuration, grid: "dict[Parameter, ParameterValuesType]"):
    tuner = GridSearchTuner(parameters=grid)
    tuner.tune(
        config.estimator,
        config.x_samples,
        config.xp_samples,
        LinearTruncatedFourierFeatureMap,
        config.num_frequencies,
        config.X_bounds,
    )
    score_estimator(config)


def hp_tuning(file_path: str, grid: "dict[Parameter, ParameterValuesType]"):
    action = ConfigAction(option_strings=None, dest="input")
    config = Configuration()
    action(None, config, Path(file_path), None)
    log.info(f"Configuration: {config}")
    if config.seed >= 0:
        np.random.seed(config.seed)  # For reproducibility
        random.seed(config.seed)
    if config.system_dynamics and len(config.x_samples) == 0 and len(config.xp_samples) == 0:
        log.info("Sampling x_samples and xp_samples from bounds")
        f = lambda x: config.system_dynamics(x) + (np.random.normal(scale=config.noise_scale))
        config.x_samples = config.X_bounds.sample(config.num_samples)
        config.xp_samples = f(config.x_samples)
    config.estimator = config.estimator(
        kernel=config.kernel(sigma_l=config.sigma_l, sigma_f=config.sigma_f),
        regularization_constant=config.lambda_,
    )

    assert_or_raise(len(config.x_samples) > 0, "No samples to use for the scenario")
    assert_or_raise(len(config.xp_samples) > 0, "No transition samples to use for the scenario")
    log.info("Initial")
    score_estimator(config)
    log.info("LBFGS Tuner")
    lbfgs_tuner(config)
    log.info("Grid Tuner")
    grid_search_tuner(config, grid)
