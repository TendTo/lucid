import numpy as np
from pylucid import *
from pylucid import __version__
from pylucid.pipeline import pipeline


def test_barrier3():
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Script configuration
    # ---------------------------------- #

    # set_verbosity(LOG_DEBUG)  # Uncomment to enable debug logging
    seed = 42  # Seed for reproducibility

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # System dynamics
    # ---------------------------------- #

    f_det = None  # lambda x: x
    # Add process noise
    np.random.seed(seed)  # For reproducibility
    f = lambda x: f_det(x) + (np.random.standard_normal())

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Safety specification
    # ---------------------------------- #

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

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Parameters and inputs
    # ---------------------------------- #

    x_samples = read_matrix("tests/bindings/pylucid/x_samples.matrix")
    xp_samples = read_matrix("tests/bindings/pylucid/xp_samples.matrix")

    # Initial estimator hyperparameters. Can be tuned later
    regularization_constant = 1e-6
    sigma_f = 19.456
    sigma_l = np.array([1.598547, 0.868538])

    num_freq_per_dim = 8  # Number of frequencies per dimension. Includes the zero frequency.

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Lucid
    # ---------------------------------- #

    # De-comment the tuner you want to use or leave it empty to avoid tuning.
    tuner = {
        # "tuner": LbfgsTuner(bounds=((0.1, 15.0),), parameters=LbgsParameters(min_step=0, linesearch=5))
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

    log_info(
        f"Estimetor parameters: sigma_l = {estimator.get(Parameter.SIGMA_L)}, "
        f"sigma_f = {estimator.get(Parameter.SIGMA_F)}, "
        f"reg = {estimator.get(Parameter.REGULARIZATION_CONSTANT)}"
    )

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
    # Running the pipeline
    # ---------------------------------- #

    pipeline(
        x_samples=x_samples,
        xp_samples=xp_samples,
        x_bounds=X_bounds,
        x_init=X_init,
        x_unsafe=X_unsafe,
        T=T,
        gamma=gamma,
        f_det=f_det,  # The deterministic part of the system dynamics
        num_freq_per_dim=num_freq_per_dim,  # Number of frequencies per dimension for the Fourier feature map
        estimator=estimator,  # The estimator used to model the system dynamics
        sigma_f=estimator.get(Parameter.SIGMA_F),
        problem_log_file="problem.lp",  # The lp file containing the optimization problem
        iis_log_file="iis.ilp",  # The ilp file containing the irreducible infeasible set (IIS) if the problem is infeasible
    )


if __name__ == "__main__":
    import time

    log_info(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()
    test_barrier3()
    end = time.time()
    log_info(f"Elapsed time: {end - start}")
