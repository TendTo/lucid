import numpy as np
from ._pylucid import (
    GaussianKernel,
    FeatureMap,
    ConstantTruncatedFourierFeatureMap,
    Set,
    Tuner,
    KernelRidgeRegressor,
    GurobiLinearOptimiser,
    LucidNotSupportedException,
    GUROBI_BUILD,
    Estimator,
    log_debug,
    log_error,
    log_warn,
)

try:
    from .dreal import verify_barrier_certificate
except ImportError:
    log_warn("Verification disabled")

    def verify_barrier_certificate(*args, **kwargs):
        pass


try:
    from .plot import plot_solution
except ImportError:
    log_warn("Plotting disabled")

    def plot_solution(*args, **kwargs):
        pass


from typing import Callable


def rmse(x: "np.typing.NDArray[np.float64]", y: "np.typing.NDArray[np.float64]", ax=0):
    return np.sqrt(((x - y) ** 2).mean(axis=ax))


def pipeline(
    x_samples: np.typing.NDArray[np.float64],
    xp_samples: np.typing.NDArray[np.float64],
    x_bounds: Set,
    x_init: Set,
    x_unsafe: Set,
    *,
    T: int = 5,
    gamma: float = 1.0,
    f_det: "Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]",
    estimator: "Estimator | None" = None,
    tuner: "Tuner | None" = None,
    num_freq_per_dim: int = 8,
    feature_map: type[FeatureMap] = ConstantTruncatedFourierFeatureMap,
    sigma_l: "float | np.typing.NDArray[np.float64]" = np.array([1.0]),
    sigma_f: float = 1.0,
    regularization_constant: float = 1e-6,
) -> None:
    """Run Lucid with the given parameters.
    This function makes it easier to work with the library by providing
    reasonable defaults and a simple interface,
    while being flexible enough to accomodate most use cases.
    If you need more control, use the individual functions and classes directly.

    Args:

    """
    assert x_samples.shape[0] == xp_samples.shape[0], "x_samples and xp_samples must have the same number of samples"
    assert (
        isinstance(sigma_l, float) or sigma_l.shape[0] == x_bounds.dimension
    ), "sigma_l must be a float or an array with the same number of elements as the number of dimensions in x_bounds"
    assert isinstance(sigma_f, float), "sigma_f must be a float"

    if tuner is None:
        tuner = None
    if estimator is None:
        estimator = KernelRidgeRegressor(
            kernel=GaussianKernel(sigma_l=sigma_l, sigma_f=sigma_f),
            regularization_constant=regularization_constant,
            tuner=tuner,
        )

    samples_per_dim = 2 * num_freq_per_dim
    n_per_dim = samples_per_dim * 2

    tffm = feature_map(num_freq_per_dim, sigma_l, sigma_f, x_bounds)

    f_xp_samples = tffm(xp_samples)  # Used to train the f_xp regressor

    estimator.fit(x=x_samples, y=f_xp_samples)

    log_debug(f"RMSE on f_xp_samples {rmse(estimator(x_samples), f_xp_samples)}")
    log_debug(f"Score on f_xp_samples {estimator.score(x_samples, f_xp_samples)}")

    x_lattice = x_bounds.lattice(n_per_dim, True)
    u_f_x_lattice = tffm(x_lattice)
    u_f_xp_lattice_via_regressor = estimator(x_lattice)  # What we want to do
    # We are fixing the zero frequency to the constant value we computed in the feature map
    # If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
    u_f_xp_lattice_via_regressor[:, 0] = tffm.weights[0]

    x0_lattice = x_init.lattice(n_per_dim, True)
    f_x0_lattice = tffm(x0_lattice)

    xu_lattice = x_unsafe.lattice(n_per_dim, True)
    f_xu_lattice = tffm(xu_lattice)

    o = GurobiLinearOptimiser(T, gamma, 0, 1, b_kappa=1, sigma_f=sigma_f)

    def check_cb(
        success: bool, obj_val: float, sol: "np.typing.NDArray[np.float64]", eta: float, c: float, norm: float
    ):
        if not success:
            log_error("Optimization failed")
            return

        log_debug(f"sol = {sol}")
        plot_solution(
            X_bounds=x_bounds,
            X_init=x_init,
            X_unsafe=x_unsafe,
            feature_map=tffm,
            eta=eta,
            gamma=gamma,
            sol=sol,
            f=f_det,
            estimator=estimator,
        )
        verify_barrier_certificate(
            X_bounds=x_bounds,
            X_init=x_init,
            X_unsafe=x_unsafe,
            sigma_f=sigma_f,
            eta=eta,
            c=c,
            f_det=f_det,
            gamma=gamma,
            estimator=estimator,
            tffm=tffm,
            sol=sol,
        )

    try:
        o.solve(
            f0_lattice=f_x0_lattice,
            fu_lattice=f_xu_lattice,
            phi_mat=u_f_x_lattice,
            w_mat=u_f_xp_lattice_via_regressor,
            rkhs_dim=tffm.dimension,
            num_frequencies_per_dim=num_freq_per_dim - 1,
            num_frequency_samples_per_dim=n_per_dim,
            original_dim=x_bounds.dimension,
            callback=check_cb,
        )
        assert GUROBI_BUILD
    except LucidNotSupportedException:
        assert not GUROBI_BUILD  # Did not compile against Gurobi. Ignore this test.
