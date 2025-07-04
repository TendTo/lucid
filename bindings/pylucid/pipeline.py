from typing import TYPE_CHECKING

import numpy as np

from ._pylucid import (
    GUROBI_BUILD,
    AlglibOptimiser,
    Estimator,
    FeatureMap,
    GaussianKernel,
    GurobiOptimiser,
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
    from .cli import Configuration


def rmse(x: "NMatrix", y: "NMatrix", ax=0):
    return np.sqrt(((x - y) ** 2).mean(axis=ax))


def tune() -> "Estimator":
    """Tune the default estimator using the median heuristic."""
    log.info("Tuning the default estimator using the median heuristic.")
    estimator = KernelRidgeRegressor(
        kernel=GaussianKernel(sigma_l=1, sigma_f=1.0),
        regularization_constant=1e-6,
        tuner=MedianHeuristicTuner(),
    )
    return estimator


def pipeline(args: "Configuration") -> bool:
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
        num_frequencies: Number of frequencies per dimension for the feature map
        oversample_factor: Factor by which to oversample the frequency space
        num_oversample: Number of samples to use for the frequency space. If negative, it is computed based on the oversample_factor
        feature_map: Feature map class to use for transformation or a callable that returns a feature map based on the estimator
        sigma_f: Signal variance parameter for the kernel
        verify: Whether to verify the barrier certificate using dReal
        plot: Whether to plot the solution using matplotlib

    Raises:
        AssertionError: If the input samples do not match in size or if sigma_f is not a float.

    Returns:
        True if the optimization was successful, False otherwise.
    """
    assert (
        args.x_samples.shape[0] == args.xp_samples.shape[0]
    ), "x_samples and xp_samples must have the same number of samples"
    assert isinstance(args.sigma_f, float) and args.sigma_f > 0, "sigma_f must be a positive float"
    assert (
        not isinstance(args.feature_map, FeatureMap) or args.num_frequencies <= 0
    ), "num_frequencies and feature_map are mutually exclusive"
    assert (
        args.f_xp_samples is not None
        or args.feature_map is None
        or isinstance(args.feature_map, FeatureMap)
        or isinstance(args.feature_map, type)
    ), "f_xp_samples must be provided when feature_map is a callback"

    if isinstance(args.estimator, type):
        estimator = args.estimator(
            kernel=args.kernel(sigma_l=1, sigma_f=args.sigma_f),
            regularization_constant=1e-6,
            tuner=MedianHeuristicTuner(),
        )
    else:
        estimator = args.estimator
    if isinstance(args.feature_map, type) and issubclass(args.feature_map, FeatureMap):
        assert args.num_frequencies > 0, "num_frequencies must be set and positive if feature_map is a class"
        sigma_l = estimator.get(Parameter.SIGMA_L)
        feature_map = args.feature_map(
            num_frequencies=args.num_frequencies,
            sigma_l=sigma_l if len(sigma_l) > 1 else sigma_l[0],
            sigma_f=args.sigma_f,
            x_limits=args.X_bounds,
        )
    else:
        feature_map = args.feature_map

    num_frequencies = feature_map.num_frequencies if args.num_frequencies < 0 else args.num_frequencies
    num_samples_per_dim = (
        np.ceil((2 * num_frequencies + 1) * args.oversample_factor) if args.num_oversample < 0 else args.num_oversample
    )
    num_samples_per_dim = int(num_samples_per_dim)
    log.debug(f"Number of samples per dimension: {num_samples_per_dim}")
    assert num_samples_per_dim > 2 * num_frequencies, "n_per_dim must be greater than nyquist (2 * num_frequencies + 1)"

    if args.f_xp_samples is None:  # If no precomputed f_xp_samples are provided, compute them
        assert isinstance(feature_map, FeatureMap), "feature_map must be a FeatureMap instance"
        args.f_xp_samples = feature_map(args.xp_samples)

    log.debug(f"Estimator pre-fit: {estimator}")
    estimator.fit(x=args.x_samples, y=args.f_xp_samples)  # Actual fitting of the regressor
    log.info(f"Estimator post-fit: {estimator}")

    if callable(feature_map) and not isinstance(feature_map, FeatureMap):
        feature_map = feature_map(estimator)  # Compute the feature map if it is a callable
    assert isinstance(feature_map, FeatureMap), "feature_map must return a FeatureMap instance"

    log.debug(f"RMSE on f_xp_samples {rmse(estimator(args.x_samples), args.f_xp_samples)}")
    log.debug(f"Score on f_xp_samples {estimator.score(args.x_samples, args.f_xp_samples)}")
    if args.system_dynamics is not None:
        # Sample some other points (half of the x_samples) to evaluate the regressor against overfitting
        x_evaluation = args.X_bounds.sample(args.x_samples.shape[0] // 2)
        f_xp_evaluation = feature_map(args.system_dynamics(x_evaluation))
        log.debug(f"RMSE on f_det_evaluated {rmse(estimator(x_evaluation), f_xp_evaluation)}")
        log.debug(f"Score on f_det_evaluated {estimator.score(x_evaluation, f_xp_evaluation)}")

    log.debug(f"Feature map: {feature_map}")
    x_lattice = args.X_bounds.lattice(num_samples_per_dim, True)
    u_f_x_lattice = feature_map(x_lattice)
    u_f_xp_lattice_via_regressor = estimator(x_lattice)
    # We are fixing the zero frequency to the constant value we computed in the feature map
    # If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
    u_f_xp_lattice_via_regressor[:, 0] = feature_map.weights[0] * args.sigma_f

    x0_lattice = args.X_init.lattice(num_samples_per_dim, True)
    f_x0_lattice = feature_map(x0_lattice)

    xu_lattice = args.X_unsafe.lattice(num_samples_per_dim, True)
    f_xu_lattice = feature_map(xu_lattice)

    def check_cb(success: bool, obj_val: float, sol: "NVector", eta: float, c: float, norm: float):
        if not success:
            log.error("Optimization failed")
        else:
            log.info("Optimization succeeded")
            log.debug(f"{obj_val = }, {eta = }, {c = }, {norm = }")
            log.debug(f"{sol = }")
        if args.plot:
            plot_solution(
                X_bounds=args.X_bounds,
                X_init=args.X_init,
                X_unsafe=args.X_unsafe,
                feature_map=feature_map,
                eta=eta if success is None else None,
                gamma=args.gamma,
                sol=sol if success else None,
                f=args.system_dynamics,
                estimator=estimator,
                num_samples=num_samples_per_dim,
                c=c if success else None,
            )
        if args.verify and args.system_dynamics is not None and success:
            verify_barrier_certificate(
                X_bounds=args.X_bounds,
                X_init=args.X_init,
                X_unsafe=args.X_unsafe,
                sigma_f=args.sigma_f,
                eta=eta,
                c=c,
                f_det=args.system_dynamics,
                gamma=args.gamma,
                estimator=estimator,
                tffm=feature_map,
                sol=sol,
            )

    files = {
        "problem_log_file": args.problem_log_file,
        "iis_log_file": args.iis_log_file,
    }

    return args.optimiser(
        args.time_horizon,
        args.gamma,
        0.0,
        1.0,
        b_kappa=1.0,
        C_coeff=args.c_coefficient,
        sigma_f=args.sigma_f,
        **(files if isinstance(args.optimiser, GurobiOptimiser) else {}),
    ).solve(
        f0_lattice=f_x0_lattice,
        fu_lattice=f_xu_lattice,
        phi_mat=u_f_x_lattice,
        w_mat=u_f_xp_lattice_via_regressor,
        rkhs_dim=feature_map.dimension,
        num_frequencies_per_dim=args.num_frequencies - 1,
        num_frequency_samples_per_dim=num_samples_per_dim,
        original_dim=args.X_bounds.dimension,
        callback=check_cb,
    )
