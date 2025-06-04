import numpy as np
from pylucid import __version__
from pylucid import *
from pylucid.pipeline import pipeline

np.set_printoptions(linewidth=200, suppress=True)

# set_verbosity(LOG_DEBUG)


def case_study_template():
    ######## System dynamics ########
    seed = 50

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

    ######## Safety specification ########

    # Time horizon
    T = 5
    # State space X := [1, 50]
    X_bounds = RectSet(((1, 50),))

    # Initial set X_I := [19.5, 20]
    X_init = RectSet(((19.5, 20),))

    # Unsafe set X_U := [1, 17] U [23, 50]
    X_unsafe = MultiSet(
        RectSet(((1, 17),)),
        RectSet(((23, 50),)),
    )

    ######## Parameters ########
    gamma = 1
    N = 1000

    x_samples = X_bounds.sample(N)
    xp_samples = f(x_samples.T).T

    # Estimator hyperparameters
    regularization_constant = 1e-6
    sigma_f = 5.0
    num_freq_per_dim = 4  # Number of frequencies per dimension. Includes the zero frequency.

    ######## Lucid ########

    params = LbgsParameters(min_step=0, linesearch=5)

    tun = {"tuner": LbfgsTuner(bounds=((0.1, 15.0),), parameters=params)}
    estimator = KernelRidgeRegressor(
        kernel=GaussianKernel(sigma_f=sigma_f),
        regularization_constant=regularization_constant,
        **tun,
    )
    # The sigma_l parameter will be tuned by the tuner during fitting
    estimator.fit(x=x_samples, y=xp_samples, **tun)

    pipeline(
        x_samples=x_samples,
        xp_samples=xp_samples,
        x_bounds=X_bounds,
        x_init=X_init,
        x_unsafe=X_unsafe,
        T=T,
        gamma=gamma,
        f_det=f_det,
        num_freq_per_dim=num_freq_per_dim,
        estimator=estimator,
        sigma_f=sigma_f,
    )


if __name__ == "__main__":
    import time

    log_info(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()
    case_study_template()
    end = time.time()
    log_info(f"Elapsed time: {end - start}")
