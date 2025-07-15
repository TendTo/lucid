from typing import TYPE_CHECKING, TypedDict

import numpy as np

from ._pylucid import (
    Estimator,
    FeatureMap,
    GaussianKernel,
    GurobiOptimiser,
    KernelRidgeRegressor,
    MedianHeuristicTuner,
    Parameter,
    log,
)

if TYPE_CHECKING:
    from typing import Callable

    from plotly.graph_objects import Figure

    from ._pylucid import NMatrix, NVector
    from .cli import Configuration


try:
    from .dreal import verify_barrier_certificate
except ImportError:
    log.warn("Verification disabled")

    def verify_barrier_certificate(*args, **kwargs) -> "bool":
        pass


try:
    from .plot import plot_solution
except ImportError:
    log.warn("Plotting disabled")

    def plot_solution(*args, **kwargs) -> "Figure":
        pass


class OptimiserResult(TypedDict):
    """Result of the optimisation process."""

    success: bool
    obj_val: float
    sol: "NVector"
    eta: float
    c: float
    norm: float
    fig: "Figure"
    verified: bool


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


def pipeline(
    args: "Configuration", show: bool = True, optimiser_cb: "Callable[[OptimiserResult], None]" = None
) -> bool:
    """Run Lucid with the given parameters.
    This function makes it easier to work with the library by providing
    reasonable defaults and a simple interface,
    while being flexible enough to accomodate most use cases.
    If you need more control, use the individual functions and classes directly.

    Args:
        args: The configuration object containing all the parameters.
        optimiser_cb: A callback function to handle the optimization results.


    Raises:
        AssertionError: If the input samples do not match in size or if sigma_f is not a float.

    Returns:
        True if the optimization was successful, False otherwise.
    """
    log.debug(f"Pipeline started with {args}")
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
            kernel=args.kernel(sigma_l=args.sigma_l, sigma_f=args.sigma_f),
            regularization_constant=args.lambda_,
            **({"tuner": args.tuner} if args.tuner is not None else {}),
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
    num_oversample = (
        np.ceil((2 * num_frequencies + 1) * args.oversample_factor) if args.num_oversample < 0 else args.num_oversample
    )
    num_oversample = int(num_oversample)
    log.debug(f"Number of samples per dimension: {num_oversample}")
    assert num_oversample > 2 * num_frequencies, "n_per_dim must be greater than nyquist (2 * num_frequencies + 1)"

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
    x_lattice = args.X_bounds.lattice(num_oversample, True)
    u_f_x_lattice = feature_map(x_lattice)
    u_f_xp_lattice_via_regressor = estimator(x_lattice)
    # We are fixing the zero frequency to the constant value we computed in the feature map
    # If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
    u_f_xp_lattice_via_regressor[:, 0] = feature_map.weights[0] * args.sigma_f

    x0_lattice = args.X_init.lattice(num_oversample, True)
    f_x0_lattice = feature_map(x0_lattice)

    xu_lattice = args.X_unsafe.lattice(num_oversample, True)
    f_xu_lattice = feature_map(xu_lattice)

    def check_cb(success: bool, obj_val: float, sol: "NVector", eta: float, c: float, norm: float):
        result = OptimiserResult(
            success=success, obj_val=obj_val, sol=sol, eta=eta, c=c, norm=norm, fig=None, verified=False
        )
        if not success:
            log.error("Optimization failed")
        else:
            log.info("Optimization succeeded")
            log.debug(f"{obj_val = }, {eta = }, {c = }, {norm = }")
            log.debug(f"{sol = }")
        if args.plot:
            result["fig"] = plot_solution(
                X_bounds=args.X_bounds,
                X_init=args.X_init,
                X_unsafe=args.X_unsafe,
                feature_map=feature_map,
                eta=eta if success else None,
                gamma=args.gamma,
                sol=sol if success else None,
                f=args.system_dynamics,
                estimator=estimator,
                num_samples=num_oversample,
                c=c if success else None,
                show=show,
            )
        if args.verify and args.system_dynamics is not None and success:
            result["verified"] = verify_barrier_certificate(
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
        if optimiser_cb is not None:
            optimiser_cb(result)

    return args.optimiser(
        args.time_horizon,
        args.gamma,
        0.0,
        1.0,
        b_kappa=1.0,
        C_coeff=args.c_coefficient,
        sigma_f=args.sigma_f,
        problem_log_file=args.problem_log_file,
        iis_log_file=args.iis_log_file,
    ).solve(
        f0_lattice=f_x0_lattice,
        fu_lattice=f_xu_lattice,
        phi_mat=u_f_x_lattice,
        w_mat=u_f_xp_lattice_via_regressor,
        rkhs_dim=feature_map.dimension,
        num_frequencies_per_dim=args.num_frequencies - 1,
        num_frequency_samples_per_dim=num_oversample,
        original_dim=args.X_bounds.dimension,
        callback=check_cb,
    )
