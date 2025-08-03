#!/usr/bin/env python3
import numpy as np

from pylucid import *
from pylucid import __version__
from pylucid.pipeline import pipeline, rmse


def scenario_config() -> "Configuration":
    # ################################## #
    # Script configuration
    # ################################## #

    seed = 42  # Seed for reproducibility

    # ################################## #
    # System dynamics
    # ################################## #

    th = 45
    te = -15
    r_coeff = 0.1
    beta = 0.06
    theta = 0.145

    # Deterministic part of the linear dynamics x[k + 1] = (1 − β − θν)x[k] + θThν + βTe + Rς
    # where ν is -0.0120155x + 0.8
    f_det = lambda x: (1 - beta - theta * -0.0120155 * x + 0.8) * x + theta * th * -0.0120155 * x + 0.8 + beta * te
    # Add process noise
    np.random.seed(seed)  # For reproducibility
    f = lambda x: f_det(x) + r_coeff * np.random.exponential(1)

    # ################################## #
    # Safety specification
    # ################################## #

    gamma = 1
    T = 5  # Time horizon

    # State space X := [1, 50]
    X_bounds = RectSet(((1, 50),))
    # Initial set X_0 := [19.5, 20]
    X_init = RectSet(((19.5, 20),))
    # Unsafe set X_U := [1, 17] U [23, 50]
    X_unsafe = MultiSet(
        RectSet(((1, 17),)),
        RectSet(((23, 50),)),
    )

    # ################################## #
    # Parameters and inputs
    # ################################## #

    N = 1000
    x_samples = X_bounds.sample(N)
    xp_samples = f(x_samples.T).T

    # Initial estimator hyperparameters. Can be tuned later
    regularization_constant = 1e-6
    sigma_f = 15.0
    sigma_l = np.array([1.75555556])

    num_freq_per_dim = 4  # Number of frequencies per dimension. Includes the zero frequency.

    # ################################## #
    # Lucid
    # ################################## #

    # De-comment the tuner you want to use or leave it empty to avoid tuning.
    tuner = {
        # "tuner": LbfgsTuner(bounds=((0.1, 15.0),), parameters=LbfgsParameters(min_step=0, linesearch=5))
        # "tuner": MedianHeuristicTuner(),
        # "tuner": GridSearchTuner(
        #     ParameterValues(
        #         Parameter.SIGMA_L, [np.full(1, v) for v in np.linspace(0.1, 15.0, num=10, endpoint=True, dtype=float)]
        #     ),
        #     ParameterValues(Parameter.SIGMA_F, np.linspace(0.1, 15.0, num=10, endpoint=True, dtype=float)),
        #     ParameterValues(Parameter.REGULARIZATION_CONSTANT, np.logspace(-6, -1, num=10)),
        # ),
    }
    estimator = KernelRidgeRegressor(
        kernel=GaussianKernel(sigma_f=sigma_f, sigma_l=sigma_l),
        regularization_constant=regularization_constant,
    )
    # Depending on the tuner selected in the dictionary above, the estimator will be fitted with different parameters.
    estimator.fit(x=x_samples, y=xp_samples, **tuner)
    log_debug(f"RMSE on xp_samples {rmse(estimator(x_samples), xp_samples)}")
    log_debug(f"Score on xp_samples {estimator.score(x_samples, xp_samples)}")

    # ################################## #
    # Running the pipeline
    # ################################## #

    return Configuration(
        x_samples=x_samples,
        xp_samples=xp_samples,
        X_bounds=X_bounds,
        X_init=X_init,
        X_unsafe=X_unsafe,
        time_horizon=T,
        gamma=gamma,
        system_dynamics=f_det,  # The deterministic part of the system dynamics
        num_frequencies=num_freq_per_dim,  # Number of frequencies per dimension for the Fourier feature map
        estimator=estimator,  # The estimator used to model the system dynamics
        sigma_f=estimator.get(Parameter.SIGMA_F),
        problem_log_file="problem.mps",  # The lp file containing the optimization problem
        iis_log_file="iis.ilp",  # The ilp file containing the irreducible infeasible set (IIS) if the problem is infeasible
    )


if __name__ == "__main__":
    # ################################## #
    # Lucid
    # ################################## #
    log.info(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()
    pipeline(scenario_config())
    end = time.time()
    log.info(f"Elapsed time: {end - start}")
