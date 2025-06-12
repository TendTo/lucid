#!/usr/bin/env python3
import numpy as np
from pylucid import *
from pylucid import __version__
from pylucid.pipeline import pipeline


def scenario_config() -> "ScenarioConfig":
    """Benchmark scenario taken from
    https://github.com/oxford-oxcav/fossil/blob/10f1f071784d16b2a5ee5da2f51ff2a81d753e2e/experiments/benchmarks/models.py#L350C1-L360C1
    """
    # ################################## #
    # Script configuration
    # ################################## #

    seed = 42  # Seed for reproducibility

    # ################################## #
    # System dynamics
    # ################################## #

    f_det = lambda x: np.array([x[:, 1], -x[:, 0] - x[:, 1] + 1 / 3 * x[:, 0] ** 3]).T  # lambda x: x
    # Add process noise
    np.random.seed(seed)  # For reproducibility
    f = lambda x: f_det(x) + (np.random.standard_normal())

    # ################################## #
    # Safety specification
    # ################################## #

    gamma = 18.312
    T = 10  # Time horizon

    X_bounds = RectSet((-3, -2), (2.5, 1), seed=seed)  # State space X
    # Initial set X_0
    X_init = MultiSet(
        RectSet((1, -0.5), (2, 0.5)),
        RectSet((-1.8, -0.1), (-1.2, 0.1)),
        RectSet((-1.4, -0.5), (-1.2, 0.1)),
    )
    # Unsafe set X_U
    X_unsafe = MultiSet(RectSet((0.4, 0.1), (0.6, 0.5)), RectSet((0.4, 0.1), (0.8, 0.3)))

    # ################################## #
    # Parameters and inputs
    # ################################## #

    N = 10
    x_samples = read_matrix("tests/bindings/pylucid/x_samples.matrix")
    xp_samples = f_det(x_samples)

    # Initial estimator hyperparameters. Can be tuned later
    regularization_constant = 1e-6
    sigma_f = 15.0
    sigma_l = np.array([1, 1.0])

    num_freq_per_dim = 4  # Number of frequencies per dimension. Includes the zero frequency.

    # ################################## #
    # Lucid
    # ################################## #

    # De-comment the tuner you want to use or leave it empty to avoid tuning.
    tuner = {
        # "tuner": LbfgsTuner(bounds=((1e-5, 1e5), (1e-5, 1e5)), parameters=LbgsParameters(min_step=0, linesearch=5))
        # "tuner": MedianHeuristicTuner(),
        # "tuner": GridSearchTuner(
        #     ParameterValues(
        #         Parameter.SIGMA_L, [np.full(2, v) for v in np.linspace(0.1, 15.0, num=10, endpoint=True, dtype=float)]
        #     ),
        #     ParameterValues(Parameter.SIGMA_F, np.linspace(0.1, 15.0, num=10, endpoint=True, dtype=float)),
        #     ParameterValues(Parameter.REGULARIZATION_CONSTANT, np.logspace(-6, -1, num=10)),
        # ),
    }
    # estimator = KernelRidgeRegressor(
    #     kernel=GaussianKernel(sigma_f=sigma_f, sigma_l=sigma_l),
    #     regularization_constant=regularization_constant,
    # )
    feature_map = ConstantTruncatedFourierFeatureMap(
        num_frequencies=num_freq_per_dim,
        sigma_l=sigma_l,
        sigma_f=sigma_f,
        x_limits=X_bounds,
    )
    print(f"Feature map: {feature_map(f_det(x_samples))}")
    estimator = ModelEstimator(f=lambda x: feature_map(f_det(x)))  # Use the custom model estimator

    # ################################## #
    # Running the pipeline
    # ################################## #

    return ScenarioConfig(
        x_samples=x_samples,
        xp_samples=xp_samples,
        X_bounds=X_bounds,
        X_init=X_init,
        X_unsafe=X_unsafe,
        T=T,
        gamma=gamma,
        f_det=f_det,  # The deterministic part of the system dynamics
        # num_freq_per_dim=num_freq_per_dim,  # Number of frequencies per dimension for the Fourier feature map
        estimator=estimator,  # The estimator used to model the system dynamics
        sigma_f=sigma_f,
        problem_log_file="problem.lp",  # The lp file containing the optimization problem
        iis_log_file="iis.ilp",  # The ilp file containing the irreducible infeasible set (IIS) if the problem is infeasible
        feature_map=feature_map,  # The feature map used to transform the state space
    )


if __name__ == "__main__":
    # ################################## #
    # Lucid
    # ################################## #
    log_info(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()
    pipeline(**scenario_config())
    end = time.time()
    log_info(f"Elapsed time: {end - start}")
