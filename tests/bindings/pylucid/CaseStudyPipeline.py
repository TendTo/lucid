import numpy as np
from pylucid import (
    __version__,
    ConstantTruncatedFourierFeatureMap,
    RectSet,
    MultiSet,
    log_info,
    set_verbosity,
    LOG_DEBUG,
    LbfgsTuner
)
from pylucid.pipeline import pipeline

np.set_printoptions(linewidth=200, suppress=True)

set_verbosity(LOG_DEBUG)


def case_study_template():
    ######## System dynamics ########

    f_det = lambda x: 1 / 2 * x
    # Add process noise
    np.random.seed(50)  # For reproducibility
    f = lambda x: f_det(x) * (np.random.standard_normal())

    ######## Safety specification ########

    # Time horizon
    T = 5
    # State space X
    X_bounds = RectSet(((-1, 1),), seed=50)

    # Initial set X_I
    X_init = RectSet(((-0.5, 0.5),))

    # Unsafe set X_U
    X_unsafe = MultiSet(
        RectSet(((-1, -0.9),)),
        RectSet(((0.9, 1),)),
    )

    ######## Parameters ########
    gamma = 1
    N = 1000

    x_samples = X_bounds.sample(N)
    xp_samples = f(x_samples.T).T

    # Estimator hyperparameters
    regularization_constant = 1e-6
    sigma_f, sigma_l = 1.0, np.array([0.5])
    num_freq_per_dim = 4  # Number of frequencies per dimension. Includes the zero frequency.

    ######## Lucid ########

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
        feature_map=ConstantTruncatedFourierFeatureMap,
        sigma_l=sigma_l,
        sigma_f=sigma_f,
        regularization_constant=regularization_constant,
        tuner=LbfgsTuner()
    )


if __name__ == "__main__":
    import time

    log_info(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()
    case_study_template()
    end = time.time()
    log_info(f"Elapsed time: {end - start}")
