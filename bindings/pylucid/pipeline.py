from typing import TYPE_CHECKING, TypedDict

import numpy as np

from ._pylucid import (
    Estimator,
    FeatureMap,
    FourierBarrierCertificate,
    GaussianKernel,
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
    from .dreal import verify_barrier_conditions
except ImportError:
    log.warn("Verification disabled")

    def verify_barrier_conditions(*args, **kwargs) -> "bool":
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
    time: float


def rmse(x: "NMatrix", y: "NMatrix", ax=0):
    return np.sqrt(((x - y) ** 2).mean(axis=ax))


def mape(x: "NMatrix", y: "NMatrix", ax=0):
    return (np.abs((x - y) / y).mean(axis=ax)) * 100


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
    config: "Configuration",
    show: bool = True,
    optimiser_cb: "Callable[[OptimiserResult], None]" = None,
    plot_cb: "Callable[[Figure], None]" = None,
    verify_cb: "Callable[[bool], None]" = None,
) -> bool:
    """Run Lucid with the given parameters.
    This function makes it easier to work with the library by providing
    reasonable defaults and a simple interface,
    while being flexible enough to accomodate most use cases.
    If you need more control, use the individual functions and classes directly.

    Args:
        config: The configuration object containing all the parameters.
        optimiser_cb: A callback function to handle the optimization results.


    Raises:
        AssertionError: If the input samples do not match in size or if sigma_f is not a float.

    Returns:
        True if the optimization was successful, False otherwise.
    """
    log.debug(f"Pipeline started with {config}")
    assert (
        config.x_samples.shape[0] == config.xp_samples.shape[0]
    ), "x_samples and xp_samples must have the same number of samples"
    assert isinstance(config.sigma_f, float) and config.sigma_f > 0, "sigma_f must be a positive float"
    assert (
        not isinstance(config.feature_map, FeatureMap) or config.num_frequencies <= 0
    ), "num_frequencies and feature_map are mutually exclusive"
    assert (
        config.f_xp_samples is not None
        or config.feature_map is None
        or isinstance(config.feature_map, (FeatureMap, type))
    ), "f_xp_samples must be provided when feature_map is a callback"

    if isinstance(config.estimator, type):
        estimator = config.estimator(
            kernel=config.kernel(sigma_l=config.sigma_l, sigma_f=config.sigma_f),
            regularization_constant=config.lambda_,
            **({"tuner": config.tuner} if config.tuner is not None else {}),
        )
    else:
        estimator = config.estimator
    if isinstance(config.feature_map, type) and issubclass(config.feature_map, FeatureMap):
        assert config.num_frequencies > 0, "num_frequencies must be set and positive if feature_map is a class"
        sigma_l = estimator.get(Parameter.SIGMA_L)
        feature_map = config.feature_map(
            num_frequencies=config.num_frequencies,
            sigma_l=sigma_l if len(sigma_l) > 1 else sigma_l[0],
            sigma_f=config.sigma_f,
            X_bounds=config.X_bounds,
        )
    else:
        feature_map = config.feature_map

    num_frequencies = feature_map.num_frequencies if config.num_frequencies < 0 else config.num_frequencies
    num_oversample = (
        np.ceil((2 * num_frequencies + 1) * config.oversample_factor)
        if config.num_oversample < 0
        else config.num_oversample
    )
    num_oversample = int(num_oversample)
    log.debug(f"Number of samples per dimension: {num_oversample}")
    assert num_oversample > 2 * num_frequencies, f"n_per_dim must be greater than nyquist ({2 * num_frequencies + 1})"

    if config.f_xp_samples is None:  # If no precomputed f_xp_samples are provided, compute them
        assert isinstance(feature_map, FeatureMap), "feature_map must be a FeatureMap instance"
        config.f_xp_samples = feature_map(config.xp_samples)

    log.debug(f"Estimator pre-fit: {estimator}")
    estimator.fit(x=config.x_samples, y=config.f_xp_samples)  # Actual fitting of the regressor
    log.info(f"Estimator post-fit: {estimator}")

    if callable(feature_map) and not isinstance(feature_map, FeatureMap):
        feature_map = feature_map(estimator)  # Compute the feature map if it is a callable
    assert isinstance(feature_map, FeatureMap), "feature_map must return a FeatureMap instance"

    log.debug(f"RMSE on f_xp_samples {rmse(estimator(config.x_samples), config.f_xp_samples)}")
    log.debug(f"Score on f_xp_samples {estimator.score(config.x_samples, config.f_xp_samples)}")
    if config.system_dynamics is not None:
        # Sample some other points (half of the x_samples) to evaluate the regressor against overfitting
        x_evaluation = config.X_bounds.sample(config.x_samples.shape[0] // 2)
        f_xp_evaluation = feature_map(config.system_dynamics(x_evaluation))
        log.debug(f"RMSE on f_det_evaluated {rmse(estimator(x_evaluation), f_xp_evaluation)}")
        log.debug(f"Score on f_det_evaluated {estimator.score(x_evaluation, f_xp_evaluation)}")

    log.debug(f"Feature map: {feature_map}")
    x_lattice = config.X_bounds.lattice(num_oversample, True)
    u_f_x_lattice = feature_map(x_lattice)
    u_f_xp_lattice_via_regressor = estimator(x_lattice)
    # We are fixing the zero frequency to the constant value we computed in the feature map
    # If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
    u_f_xp_lattice_via_regressor[:, 0] = feature_map.weights[0] * config.sigma_f
    log.debug(f"x_lattice: {x_lattice.shape}, u_f_x_lattice: {u_f_x_lattice.shape}")

    if config.constant_lattice_points:
        x0_lattice = config.X_init.lattice(num_oversample, True)
        xu_lattice = config.X_unsafe.lattice(num_oversample, True)
    else:
        # TODO: implement this more efficiently in lucid (C++)
        # Extreme points are always included in the lattice,
        # to make sure gamma and eta conditions are satisfied on the boundaries
        x0_extreme_points = config.X_init.lattice(2, True)
        x0_lattice = np.concatenate([x0_extreme_points, np.empty_like(x_lattice)], axis=0)
        count = x0_extreme_points.shape[0]
        for point in x_lattice:
            if point in config.X_init:
                x0_lattice[count] = point
                count += 1
        x0_lattice.resize((count, config.X_bounds.dimension))

        # Extreme points are always included in the lattice,
        # to make sure gamma and eta conditions are satisfied on the boundaries
        xu_extreme_points = config.X_unsafe.lattice(2, True)
        xu_lattice = np.concatenate([xu_extreme_points, np.empty_like(x_lattice)], axis=0)
        count = xu_extreme_points.shape[0]
        for point in x_lattice:
            if point in config.X_unsafe:
                xu_lattice[count] = point
                count += 1
        xu_lattice.resize((count, config.X_bounds.dimension))
    log.debug(f"x0_lattice: {x0_lattice.shape}, xu_lattice: {xu_lattice.shape}")

    f_x0_lattice = feature_map(x0_lattice)
    f_xu_lattice = feature_map(xu_lattice)

    barrier = FourierBarrierCertificate(T=config.time_horizon, gamma=config.gamma)
    success = barrier.synthesize(
        fx_lattice=u_f_x_lattice,
        fxp_lattice=u_f_xp_lattice_via_regressor,
        fx0_lattice=f_x0_lattice,
        fxu_lattice=f_xu_lattice,
        feature_map=feature_map,
        num_frequency_samples_per_dim=num_oversample,
        c_coeff=config.c_coefficient,
        epsilon=config.epsilon,
        target_norm=config.b_norm,
        b_kappa=config.b_kappa,
        optimiser=config.optimiser(config.problem_log_file, config.iis_log_file),
    )

    obj_val = 1 - barrier.safety
    eta = barrier.eta
    c = barrier.c
    norm = barrier.norm
    sol = barrier.coefficients

    if not success:
        log.error("Optimization failed")
    else:
        log.info("Optimization succeeded")
        log.debug(f"{obj_val = }, {eta = }, {c = }, {norm = }")
        log.debug(f"{sol = }")
    if optimiser_cb is not None:
        optimiser_cb(
            OptimiserResult(
                success=success,
                obj_val=obj_val,
                sol=sol,
                eta=eta,
                c=c,
                norm=norm,
            )
        )
    if config.plot and config.X_bounds.dimension <= 2:
        log.info("Plotting the solution")
        fig = plot_solution(
            X_bounds=config.X_bounds,
            X_init=config.X_init,
            X_unsafe=config.X_unsafe,
            feature_map=feature_map,
            eta=eta if success else None,
            gamma=config.gamma,
            sol=sol if success else None,
            f=config.system_dynamics,
            estimator=estimator,
            num_samples=num_oversample,
            c=c if success else None,
            show=show,
        )
        if plot_cb is not None:
            plot_cb(fig)
    if config.verify and success:
        log.info("Verifying the solution")
        verified = verify_barrier_conditions(
            X_bounds=config.X_bounds,
            X_init=config.X_init,
            X_unsafe=config.X_unsafe,
            sigma_f=config.sigma_f,
            eta=eta,
            c=c,
            gamma=config.gamma,
            estimator=estimator,
            tffm=feature_map,
            sol=sol,
            epsilon=config.epsilon,
            b_norm=config.b_norm,
        )
        if verify_cb is not None:
            verify_cb(verified)
