import itertools
import multiprocessing
from datetime import datetime
from typing import Any, TYPE_CHECKING

import mlflow
import mlflow.data
import mlflow.entities

from pylucid import *
from pylucid.plot import plot_solution

if TYPE_CHECKING:
    from pylucid._pylucid import NMatrix


def rmse(x: "NMatrix", y: "NMatrix", ax=0) -> "np.ndarray":
    return np.sqrt(((x - y) ** 2).mean(axis=ax))


def grid_to_config(grid_keys: list[str], param_combination: list[Any]) -> Configuration:
    """Convert grid parameters to a configuration object."""
    config = Configuration()
    for key, value in zip(grid_keys, param_combination):
        setattr(config, key, value)
    return config


def benchmark(name: str, config: Configuration, grid: dict[str, list[Any]]):
    """Run the benchmark scenario."""
    logs: list[str] = []
    log.set_sink(logs.append)
    log.set_verbosity(log.LOG_DEBUG)

    config = config.shallow_copy()
    mlflow.set_experiment(experiment_name=name)
    with mlflow.start_run(run_name=f"{name}-{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"):
        mlflow.log_input(dataset=mlflow.data.from_numpy(config.x_samples, targets=config.xp_samples))
        mlflow.set_tag("scenario", name)
        i = 0
        for param_combination in itertools.product(*grid.values()):
            for key, value in zip(grid.keys(), param_combination):
                setattr(config, key, value)
            with mlflow.start_run(run_name=f"{name}-{i}", nested=True):
                try:
                    benchmark_pipeline(config=config)
                    status = mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.FINISHED)
                except Exception as ex:
                    log.error(f"Error in benchmark {name} with configuration {config.to_safe_dict()}: {ex}")
                    status = mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.FAILED)
                i += 1
                mlflow.log_text("\n".join(logs), "logs.log")
                mlflow.end_run(status=status)
            logs = []
    log.clear()


def single_benchmark(name: str, config: Configuration):
    """Run the benchmark scenario."""
    logs: list[str] = []

    def handle_log(message: str):
        logs.append(message)
        if "C:" in message:
            print(message)

    log.set_sink(handle_log)
    log.set_verbosity(log.LOG_DEBUG)

    mlflow.set_experiment(experiment_name=name)
    with mlflow.start_run(run_name=f"{name}-{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"):
        mlflow.log_input(dataset=mlflow.data.from_numpy(config.x_samples, targets=config.xp_samples))
        mlflow.set_tag("scenario", name)
        try:
            benchmark_pipeline(config=config)
            with Stats() as stats:
                stats.collect_peak_rss_memory_usage()
                mlflow.log_metric("peak_rss_memory_usage_bytes", stats.peak_rss_memory_usage)
            status = mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.FINISHED)
        except Exception as ex:
            log.error(f"Error in benchmark {name} with configuration {config.to_safe_dict()}: {ex}")
            status = mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.FAILED)
        finally:
            log.clear()
        mlflow.log_text("\n".join(logs), "logs.log")
        mlflow.end_run(status=status)


def _run_single_benchmark_with_params_factory(config: Configuration):
    """Factory function to create a helper function for running a single benchmark with specific parameters."""

    def _run_single_benchmark_with_params(values):
        """Run a single benchmark with the given parameters."""
        # Create a copy of the configuration and apply the parameter combination
        name, grid_keys, param_combination, run_index = values
        config_copy = config.shallow_copy()
        for key, value in zip(grid_keys, param_combination):
            setattr(config_copy, key, value)

        # Create a unique name for this parameter combination
        param_str = "_".join([f"{k}={v}" for k, v in zip(grid_keys, param_combination)])
        run_name = f"{name}_run{run_index}_{param_str}"

        # Run the single benchmark
        single_benchmark(run_name, config_copy)

    return _run_single_benchmark_with_params


def multi_benchmark(name: str, config: Configuration, grid: dict[str, list[Any]]):
    """Run the benchmark scenario with each grid combination in a separate process."""
    # Generate all parameter combinations
    param_combinations = list(itertools.product(*grid.values()))
    grid_keys = list(grid.keys())

    # Prepare arguments for multiprocessing
    args_list = [(name, grid_keys, param_combination, i) for i, param_combination in enumerate(param_combinations)]

    # Run benchmarks in parallel using multiprocessing
    with multiprocessing.Pool() as pool:
        pool.map(_run_single_benchmark_with_params_factory(config), args_list)


class TimeLogger:
    """Context manager to log the time taken by a block of code."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = -1

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        mlflow.log_metric(f"duration.{self.name}", (datetime.now() - self.start_time).total_seconds())


def benchmark_pipeline(config: Configuration):
    with TimeLogger("setup"):
        config_dict = config.to_safe_dict()
        mlflow.log_dict(config_dict, "config.yaml")
        for key, value in config_dict.items():
            if key not in ("f_xp_samples", "x_samples", "xp_samples"):
                mlflow.log_param(key, value)

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
            feature_map = config.feature_map(
                num_frequencies=config.num_frequencies,
                sigma_l=config.feature_sigma_l,
                sigma_f=config.sigma_f,
                X_bounds=config.X_bounds,
            )
        else:
            feature_map = config.feature_map

        num_frequencies = feature_map.num_frequencies if config.num_frequencies < 0 else config.num_frequencies
        lattice_resolution = (
            np.ceil((2 * num_frequencies + 1) * config.oversample_factor)
            if config.lattice_resolution < 0
            else config.lattice_resolution
        )
        lattice_resolution = int(lattice_resolution)
        log.debug(f"Number of samples per dimension: {lattice_resolution}")
        assert (
            lattice_resolution > 2 * num_frequencies
        ), "n_per_dim must be greater than nyquist (2 * num_frequencies + 1)"

        if config.f_xp_samples is None:  # If no precomputed f_xp_samples are provided, compute them
            assert isinstance(feature_map, FeatureMap), "feature_map must be a FeatureMap instance"
            config.f_xp_samples = feature_map(config.xp_samples)

    with TimeLogger("fit"):
        log.debug(f"Estimator pre-fit: {estimator}")
        estimator.fit(x=config.x_samples, y=config.f_xp_samples)  # Actual fitting of the regressor
        log.info(f"Estimator post-fit: {estimator}")

    with TimeLogger("evaluate"):
        if callable(feature_map) and not isinstance(feature_map, FeatureMap):
            feature_map = feature_map(estimator)  # Compute the feature map if it is a callable
        assert isinstance(feature_map, FeatureMap), "feature_map must return a FeatureMap instance"
        for i, val in enumerate(rmse(estimator(config.x_samples), config.f_xp_samples)):
            mlflow.log_metric(f"f_xp_samples.rmse.{i}", val)
        mlflow.log_metric("f_xp_samples.score", estimator.score(config.x_samples, config.f_xp_samples))
        if config.system_dynamics is not None:
            # Sample some other points (half of the x_samples) to evaluate the regressor against overfitting
            x_evaluation = config.X_bounds.sample(config.x_samples.shape[0] // 2)
            f_xp_evaluation = feature_map(config.system_dynamics(x_evaluation))
            # for i, val in enumerate(rmse(estimator(x_evaluation), f_xp_evaluation)):
            #     mlflow.log_metric(f"f_xp_evaluation.rmse.{i}", val)
            mlflow.log_metric("f_xp_evaluation.score", estimator.score(x_evaluation, f_xp_evaluation))

    with TimeLogger("solve"):
        optimiser: Optimiser = config.optimiser(
            problem_log_file=config.problem_log_file,
            iis_log_file=config.iis_log_file,
        )
        b = FourierBarrierCertificate(T=config.time_horizon, gamma=config.gamma)
        success = b.synthesize(
            optimiser=optimiser,
            lattice_resolution=config.lattice_resolution,
            estimator=estimator,
            feature_map=feature_map,
            X_bounds=config.X_bounds,
            X_init=config.X_init,
            X_unsafe=config.X_unsafe,
            parameters=FourierBarrierCertificateParameters(
                set_scaling=config.set_scaling,
                b_norm=config.b_norm,
                epsilon=config.epsilon,
                kappa=config.b_kappa,
                C_coeff=config.C_coeff,
            ),
        )
    check_cb_factory(
        success=success,
        config=config,
        lattice_resolution=lattice_resolution,
        feature_map=feature_map,
        estimator=estimator,
    )(b)
    return success


def check_cb_factory(
    success: bool, config: Configuration, lattice_resolution: int, feature_map: FeatureMap, estimator: Estimator
):

    def check_cb(b: FourierBarrierCertificate):
        mlflow.log_metrics(
            {
                "run.success": success,
                "run.safety": b.safety,
                "run.eta": b.eta,
                "run.c": b.c,
                "run.norm": b.norm,
            }
        )
        b.coefficients.shape
        if success:
            mlflow.log_table({"solution": b.coefficients.tolist()}, "solution.json")
        if config.plot and config.X_bounds.dimension <= 2:
            fig = plot_solution(
                X_bounds=config.X_bounds,
                X_init=config.X_init,
                X_unsafe=config.X_unsafe,
                feature_map=feature_map,
                eta=b.eta if success else None,
                gamma=config.gamma,
                sol=b.coefficients if success else None,
                f=config.system_dynamics,
                estimator=estimator,
                num_samples=lattice_resolution,
                c=b.c if success else None,
                show=False,
            )
            mlflow.log_figure(
                fig,
                "solution.html",
            )
        if config.verify and config.system_dynamics is not None and success:
            try:
                from pylucid.dreal import verify_barrier_certificate
            except ImportError:
                log.warn("Verification disabled")

                def verify_barrier_certificate(*args, **kwargs) -> "bool":
                    return False

            mlflow.log_metric(
                "run.verified",
                verify_barrier_certificate(
                    X_bounds=config.X_bounds,
                    X_init=config.X_init,
                    X_unsafe=config.X_unsafe,
                    sigma_f=config.sigma_f,
                    eta=b.eta,
                    c=b.c,
                    f_det=config.system_dynamics,
                    gamma=config.gamma,
                    estimator=estimator,
                    tffm=feature_map,
                    sol=b.coefficients,
                ),
            )

    return check_cb
