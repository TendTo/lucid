import numpy as np
from ._pylucid import (
    GaussianKernel,
    FeatureMap,
    ConstantTruncatedFourierFeatureMap,
    Set,
    MedianHeuristicTuner,
    KernelRidgeRegressor,
    GurobiLinearOptimiser,
    LucidNotSupportedException,
    GUROBI_BUILD,
    Estimator,
    log_debug,
    log_error,
    log_warn,
    Parameter,
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
    f_det: "Callable[[np.typing.NDArray[np.float64]], np.typing.NDArray[np.float64]]" = None,
    estimator: "Estimator | None" = None,
    num_freq_per_dim: int = -1,
    feature_map: FeatureMap = None,
    sigma_f: float = 1.0,
    verify: bool = True,
    plot: bool = True,
    problem_log_file: str = "",
    iis_log_file: str = "",
) -> None:
    """Run Lucid with the given parameters.
    This function makes it easier to work with the library by providing
    reasonable defaults and a simple interface,
    while being flexible enough to accomodate most use cases.
    If you need more control, use the individual functions and classes directly.

    Args:
        x_samples: Input samples for the state variable x
        xp_samples: Input samples for the next state variable x'
        x_bounds: Set representing the bounds of the state space
        x_init: Set representing the initial states
        x_unsafe: Set representing the unsafe states
        T: Time horizon for the optimization
        gamma: Discount or scaling factor for the optimization
        f_det: Deterministic function mapping states to outputs. Used to verify the barrier certificate.
        estimator: Estimator object for regression. If None, a default KernelRidgeRegressor is used
        num_freq_per_dim: Number of frequencies per dimension for the feature map
        feature_map: Feature map class to use for transformation
        sigma_f: Signal variance parameter for the kernel
        verify: Whether to verify the barrier certificate using dReal
        plot: Whether to plot the solution using matplotlib

    Raises:
        AssertionError: If the input samples do not match in size or if sigma_f is not a float.
    """
    assert x_samples.shape[0] == xp_samples.shape[0], "x_samples and xp_samples must have the same number of samples"
    assert isinstance(sigma_f, float) and sigma_f > 0, "sigma_f must be a positive float"
    assert feature_map is None or num_freq_per_dim <= 0, "num_freq_per_dim and feature_map are mutually exclusive"

    if estimator is None:
        estimator = KernelRidgeRegressor(
            kernel=GaussianKernel(sigma_l=1, sigma_f=sigma_f),
            regularization_constant=1e-6,
            tuner=MedianHeuristicTuner(),
        )
        estimator.fit(x=x_samples, y=xp_samples)
    if feature_map is None:
        assert num_freq_per_dim > 0, "num_freq_per_dim must be set and positive if feature_map is None"
        feature_map = ConstantTruncatedFourierFeatureMap(
            num_frequencies=num_freq_per_dim,
            sigma_l=estimator.get(Parameter.SIGMA_L),
            sigma_f=sigma_f,
            x_limits=x_bounds,
        )

    num_freq_per_dim = feature_map.num_frequencies
    samples_per_dim = 2 * num_freq_per_dim
    n_per_dim = 2 * samples_per_dim

    f_xp_samples = feature_map(xp_samples)  # Used to train the f_xp regressor

    estimator.fit(x=x_samples, y=f_xp_samples)

    log_debug(f"RMSE on f_xp_samples {rmse(estimator(x_samples), f_xp_samples)}")
    log_debug(f"Score on f_xp_samples {estimator.score(x_samples, f_xp_samples)}")

    x_lattice = x_bounds.lattice(n_per_dim, True)
    u_f_x_lattice = feature_map(x_lattice)
    u_f_xp_lattice_via_regressor = estimator(x_lattice)
    # We are fixing the zero frequency to the constant value we computed in the feature map
    # If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
    u_f_xp_lattice_via_regressor[:, 0] = feature_map.weights[0]

    x0_lattice = x_init.lattice(n_per_dim, True)
    f_x0_lattice = feature_map(x0_lattice)

    xu_lattice = x_unsafe.lattice(n_per_dim, True)
    f_xu_lattice = feature_map(xu_lattice)

    def check_cb(
        success: bool, obj_val: float, sol: "np.typing.NDArray[np.float64]", eta: float, c: float, norm: float
    ):
        if not success:
            log_error("Optimization failed")
            return

        log_debug(f"sol = {sol}")
        if verify and f_det is not None:
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
                tffm=feature_map,
                sol=sol,
            )
        if plot:
            plot_solution(
                X_bounds=x_bounds,
                X_init=x_init,
                X_unsafe=x_unsafe,
                feature_map=feature_map,
                eta=eta,
                gamma=gamma,
                sol=sol,
                f=f_det,
                estimator=estimator,
            )

    assert GUROBI_BUILD, "Gurobi is not supported in this build. Please install Gurobi and rebuild Lucid."
    o = GurobiLinearOptimiser(
        T, gamma, 0, 1, b_kappa=1, sigma_f=sigma_f, problem_log_file=problem_log_file, iis_log_file=iis_log_file
    )
    o.solve(
        f0_lattice=f_x0_lattice,
        fu_lattice=f_xu_lattice,
        phi_mat=u_f_x_lattice,
        w_mat=u_f_xp_lattice_via_regressor,
        rkhs_dim=feature_map.dimension,
        num_frequencies_per_dim=num_freq_per_dim - 1,
        num_frequency_samples_per_dim=n_per_dim,
        original_dim=x_bounds.dimension,
        callback=check_cb,
    )
