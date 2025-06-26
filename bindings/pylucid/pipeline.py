from typing import TYPE_CHECKING

import numpy as np

from ._pylucid import (
    GUROBI_BUILD,
    ConstantTruncatedFourierFeatureMap,
    Estimator,
    FeatureMap,
    GaussianKernel,
    GurobiLinearOptimiser,
    KernelRidgeRegressor,
    LinearTruncatedFourierFeatureMap,
    MedianHeuristicTuner,
    Parameter,
    Set,
    log,
)

try:
    from .dreal import verify_barrier_certificate
except ImportError:
    log.warn("Verification disabled")

    def verify_barrier_certificate(*args, **kwargs):
        pass


try:
    from .plot import plot_solution
except ImportError:
    log.warn("Plotting disabled")

    def plot_solution(*args, **kwargs):
        pass


if TYPE_CHECKING:
    from typing import Callable

    from ._pylucid import NMatrix, NVector


def rmse(x: "NMatrix", y: "NMatrix", ax=0):
    return np.sqrt(((x - y) ** 2).mean(axis=ax))


def pipeline(
    x_samples: "NMatrix",
    xp_samples: "NMatrix",
    X_bounds: Set,
    X_init: Set,
    X_unsafe: Set,
    *,
    T: int = 5,
    gamma: float = 1.0,
    c_coefficient: float = 1.0,
    f_xp_samples: "NMatrix | Callable[[Estimator, NMatrix], NMatrix] | None" = None,
    f_det: "Callable[[NMatrix], NMatrix] | None" = None,
    estimator: "Estimator | None" = None,
    num_freq_per_dim: int = -1,
    oversample_factor: float = 2.0,
    num_oversample: int = -1,
    feature_map: "FeatureMap | type[FeatureMap] | Callable[[Estimator], FeatureMap] | None" = None,
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
        X_bounds: Set representing the bounds of the state space
        X_init: Set representing the initial states
        X_unsafe: Set representing the unsafe states
        T: Time horizon for the optimization
        gamma: Discount or scaling factor for the optimization
        c_coefficient: coefficient that can be used to make the optimization more (> 1) or less (< 1) conservative
        f_xp_samples: Precomputed samples of the next state variable x' or a function that computes them
        f_det: Deterministic function mapping states to outputs. Used to verify the barrier certificate
        estimator: Estimator object for regression. If None, a default KernelRidgeRegressor is used
        num_freq_per_dim: Number of frequencies per dimension for the feature map
        oversample_factor: Factor by which to oversample the frequency space
        num_oversample: Number of samples to use for the frequency space. If negative, it is computed based on the oversample_factor
        feature_map: Feature map class to use for transformation or a callable that returns a feature map based on the estimator
        sigma_f: Signal variance parameter for the kernel
        verify: Whether to verify the barrier certificate using dReal
        plot: Whether to plot the solution using matplotlib

    Raises:
        AssertionError: If the input samples do not match in size or if sigma_f is not a float.
    """
    assert x_samples.shape[0] == xp_samples.shape[0], "x_samples and xp_samples must have the same number of samples"
    assert isinstance(sigma_f, float) and sigma_f > 0, "sigma_f must be a positive float"
    assert (
        not isinstance(feature_map, FeatureMap) or num_freq_per_dim <= 0
    ), "num_freq_per_dim and feature_map are mutually exclusive"
    assert (
        f_xp_samples is not None
        or feature_map is None
        or isinstance(feature_map, FeatureMap)
        or isinstance(feature_map, type)
    ), "f_xp_samples must be provided when feature_map is a "

    if estimator is None:
        estimator = KernelRidgeRegressor(
            kernel=GaussianKernel(sigma_l=1, sigma_f=sigma_f),
            regularization_constant=1e-6,
            tuner=MedianHeuristicTuner(),
        )
    if feature_map is None:
        assert num_freq_per_dim > 0, "num_freq_per_dim must be set and positive if feature_map is None"
        feature_map = LinearTruncatedFourierFeatureMap(
            num_frequencies=num_freq_per_dim,
            sigma_l=estimator.get(Parameter.SIGMA_L),
            sigma_f=sigma_f,
            x_limits=X_bounds,
        )
    elif isinstance(feature_map, type) and issubclass(feature_map, FeatureMap):
        assert num_freq_per_dim > 0, "num_freq_per_dim must be set and positive if feature_map is a class"
        feature_map = feature_map(
            num_frequencies=num_freq_per_dim,
            sigma_l=estimator.get(Parameter.SIGMA_L),
            sigma_f=sigma_f,
            x_limits=X_bounds,
        )

    num_freq_per_dim = feature_map.num_frequencies if num_freq_per_dim < 0 else num_freq_per_dim
    n_per_dim = np.ceil((2 * num_freq_per_dim + 1) * oversample_factor) if num_oversample < 0 else num_oversample
    n_per_dim = int(n_per_dim)
    log.debug(f"Number of samples per dimension: {n_per_dim}")
    assert n_per_dim > 2 * num_freq_per_dim, "n_per_dim must be greater than nyquist (2 * num_freq_per_dim + 1)"

    if f_xp_samples is None:  # If no precomputed f_xp_samples are provided, compute them
        assert isinstance(feature_map, FeatureMap), "feature_map must be a FeatureMap instance"
        f_xp_samples = feature_map(xp_samples)

    log.debug(f"Estimator pre-fit: {estimator}")
    estimator.fit(x=x_samples, y=f_xp_samples)  # Actual fitting of the regressor
    log.info(f"Estimator post-fit: {estimator}")

    if callable(feature_map) and not isinstance(feature_map, FeatureMap):
        feature_map = feature_map(estimator)  # Compute the feature map if it is a callable
    assert isinstance(feature_map, FeatureMap), "feature_map must return a FeatureMap instance"

    log.debug(f"RMSE on f_xp_samples {rmse(estimator(x_samples), f_xp_samples)}")
    log.debug(f"Score on f_xp_samples {estimator.score(x_samples, f_xp_samples)}")
    if f_det is not None:
        # Sample some other points (half of the x_samples) to evaluate the regressor against overfitting
        x_evaluation = X_bounds.sample(x_samples.shape[0] // 2)
        f_xp_evaluation = feature_map(f_det(x_evaluation))
        log.debug(f"RMSE on f_det_evaluated {rmse(estimator(x_evaluation), f_xp_evaluation)}")
        log.debug(f"Score on f_det_evaluated {estimator.score(x_evaluation, f_xp_evaluation)}")

    log.debug(f"Feature map: {feature_map}")
    x_lattice = X_bounds.lattice(n_per_dim, True)
    u_f_x_lattice = feature_map(x_lattice)
    u_f_xp_lattice_via_regressor = estimator(x_lattice)
    # We are fixing the zero frequency to the constant value we computed in the feature map
    # If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
    u_f_xp_lattice_via_regressor[:, 0] = feature_map.weights[0] * sigma_f

    x0_lattice = X_init.lattice(n_per_dim, True)
    f_x0_lattice = feature_map(x0_lattice)

    xu_lattice = X_unsafe.lattice(n_per_dim, True)
    f_xu_lattice = feature_map(xu_lattice)

    def check_cb(success: bool, obj_val: float, sol: "NVector", eta: float, c: float, norm: float):
        if not success:
            log.error("Optimization failed")
        else:
            log.info("Optimization succeeded")
            log.debug(f"{obj_val = }, {eta = }, {c = }, {norm = }")
            log.debug(f"{sol = }")
        if plot:
            plot_solution(
                X_bounds=X_bounds,
                X_init=X_init,
                X_unsafe=X_unsafe,
                feature_map=feature_map,
                eta=eta if success is None else None,
                gamma=gamma,
                sol=sol if success else None,
                f=f_det,
                estimator=estimator,
                c=c if success else None,
            )
        if verify and f_det is not None and success:
            verify_barrier_certificate(
                X_bounds=X_bounds,
                X_init=X_init,
                X_unsafe=X_unsafe,
                sigma_f=sigma_f,
                eta=eta,
                c=c,
                f_det=f_det,
                gamma=gamma,
                estimator=estimator,
                tffm=feature_map,
                sol=sol,
            )

    assert GUROBI_BUILD, "Gurobi is not supported in this build. Please install Gurobi and rebuild Lucid."
    o = GurobiLinearOptimiser(
        T,
        gamma,
        0,
        1,
        b_kappa=1,
        C_coeff=c_coefficient,
        sigma_f=sigma_f,
        problem_log_file=problem_log_file,
        iis_log_file=iis_log_file,
    )
    o.solve(
        f0_lattice=f_x0_lattice,
        fu_lattice=f_xu_lattice,
        phi_mat=u_f_x_lattice,
        w_mat=u_f_xp_lattice_via_regressor,
        rkhs_dim=feature_map.dimension,
        num_frequencies_per_dim=num_freq_per_dim - 1,
        num_frequency_samples_per_dim=n_per_dim,
        original_dim=X_bounds.dimension,
        callback=check_cb,
    )
