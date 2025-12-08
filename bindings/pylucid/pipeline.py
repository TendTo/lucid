from typing import TYPE_CHECKING, TypedDict

import numpy as np

from ._pylucid import (
    Estimator,
    FeatureMap,
    FourierBarrierCertificate,
    FourierBarrierCertificateParameters,
    GaussianKernel,
    KernelRidgeRegressor,
    MedianHeuristicTuner,
    ModelEstimator,
    log,
)
from .plot import plot_solution

if TYPE_CHECKING:
    from typing import Callable

    from plotly.graph_objects import Figure

    from ._pylucid import NMatrix, NVector
    from .cli import Configuration


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


def run_pipeline(
    config: "Configuration",
    lattice_resolution: int,
    estimator: "Estimator",
    feature_map: "FeatureMap",
    optimiser_cb: "Callable[[OptimiserResult], None]" = None,
    plot_cb: "Callable[[Figure], None]" = None,
    verify_cb: "Callable[[bool], None]" = None,
    show: bool = True,
) -> bool:
    """Having setup the configuration, feature map, and estimator, run the main pipeline.

    Args:
        config: The configuration object containing all the parameters.
        lattice_resolution: The resolution of the lattice to use for synthesis.
        estimator: The trained estimator to use for synthesis.
        feature_map: The feature map used to train the estimator.
        optimiser_cb: A callback function to handle the optimization results.
        plot_cb: A callback function to handle the plotting results.
        verify_cb: A callback function to handle the verification results.
        show: Whether to show the plots.

    Returns:
        True if the optimization was successful, False otherwise.
    """
    barrier = FourierBarrierCertificate(T=config.time_horizon, gamma=config.gamma)
    success = barrier.synthesize(
        lattice_resolution=lattice_resolution,
        estimator=estimator,
        X_bounds=config.X_bounds,
        X_init=config.X_init,
        X_unsafe=config.X_unsafe,
        feature_map=feature_map,
        parameters=FourierBarrierCertificateParameters(
            b_norm=config.b_norm,
            epsilon=config.epsilon,
            C_coeff=config.C_coeff,
            set_scaling=config.set_scaling,
            kappa=config.b_kappa,
        ),
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
            num_samples=lattice_resolution,
            c=c if success else None,
            show=show,
        )
        if plot_cb is not None:
            plot_cb(fig)
    if config.verify and success:
        try:
            from .dreal import verify_barrier_conditions

            log.info("Verifying the solution")
        except ImportError:
            log.warn("Verification disabled")

            def verify_barrier_conditions(*args, **kwargs) -> "bool":
                pass

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

    return success


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

    # Initialize feature map
    if isinstance(config.feature_map, type) and issubclass(config.feature_map, FeatureMap):
        assert config.num_frequencies > 0, "num_frequencies must be set and positive if feature_map is a class"
        feature_map = config.feature_map(
            num_frequencies=config.num_frequencies,
            sigma_l=config.feature_sigma_l,
            sigma_f=config.sigma_f,
            X_bounds=config.X_bounds,
        )
    else:
        feature_map = config.feature_map
    log.debug(f"Feature map: {feature_map}")

    # Initialize estimator
    if isinstance(config.estimator, type):
        if config.estimator == ModelEstimator:
            assert config.system_dynamics is not None, "system_dynamics must be provided when using ModelEstimator"
            assert isinstance(feature_map, FeatureMap), "feature_map must be a FeatureMap instance"
            estimator = ModelEstimator(lambda x: feature_map(config.system_dynamics(x)))
        else:
            estimator = config.estimator(
                kernel=config.kernel(sigma_l=config.sigma_l, sigma_f=config.sigma_f),
                regularization_constant=config.lambda_,
                **({"tuner": config.tuner} if config.tuner is not None else {}),
            )
    else:
        estimator = config.estimator

    # Determine lattice resolution
    num_frequencies = feature_map.num_frequencies if config.num_frequencies < 0 else config.num_frequencies
    lattice_resolution = (
        np.ceil((2 * num_frequencies + 1) * config.oversample_factor)
        if config.lattice_resolution < 0
        else config.lattice_resolution
    )
    lattice_resolution = int(lattice_resolution)
    log.debug(f"Number of points per dimension: {lattice_resolution}")

    assert (
        lattice_resolution > 2 * num_frequencies
    ), f"n_per_dim must be greater than nyquist ({2 * num_frequencies + 1})"

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

    return run_pipeline(
        config=config,
        lattice_resolution=lattice_resolution,
        estimator=estimator,
        feature_map=feature_map,
        optimiser_cb=optimiser_cb,
        plot_cb=plot_cb,
        verify_cb=verify_cb,
        show=show,
    )
