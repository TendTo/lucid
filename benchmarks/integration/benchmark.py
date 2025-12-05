import argparse
import itertools
import multiprocessing
import os
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.data
import mlflow.entities

from pylucid import *
from pylucid.plot import plot_solution

if TYPE_CHECKING:
    from typing import Sequence

    from pylucid._pylucid import NMatrix


class BenchmarkArgs(argparse.Namespace):
    parallel: bool
    jobs: int
    scenario: str
    single: bool


def parse_args(scenario: str = "", args: "Sequence[str] | None" = None) -> BenchmarkArgs:
    parser = argparse.ArgumentParser(description="Benchmark configuration")
    parser.add_argument("-p", "--parallel", action="store_true", help="Run benchmarks in parallel (multiprocessing)")
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(multiprocessing.cpu_count() - 2, 1),
        help="Number of parallel jobs to run (default: CPU cores - 2)",
    )
    parser.add_argument("-s", "--scenario", type=str, default=scenario, help="Scenario YAML file to run")
    parser.add_argument("--single", action="store_true", help="Only run a single benchmark (for debugging)")
    args = parser.parse_args(args, namespace=BenchmarkArgs())
    return args


def run_grid(
    args: BenchmarkArgs,
    grid: "dict[str, list[Any]]" = None,
    params_list: "tuple[tuple[tuple[str, ...], tuple[Any, ...]]]" = None,
):
    if grid is not None and params_list is not None:
        raise ValueError("Either grid or params_list should be provided, not both.")
    if grid is not None:
        param_combinations = list(itertools.product(*grid.values()))
        grid_keys = list(grid.keys())
        print(f"Running {len(param_combinations)} configurations.")
        params_list = [(grid_keys, param_combination) for param_combination in param_combinations]

    # Run benchmarks in parallel using multiprocessing
    if args.parallel and not args.single:
        with multiprocessing.Pool(processes=max(1, args.jobs)) as pool:
            pool.starmap(scenario_config, [(args.scenario,) + params for params in params_list])
    else:
        for i, params in enumerate(params_list):
            print(f"Running scenario {i+1}/{len(params_list)} with: {params}")
            scenario_config(args.scenario, *params)
            if args.single:
                break


def rmse(x: "NMatrix", y: "NMatrix", ax=0) -> "np.ndarray":
    return np.sqrt(((x - y) ** 2).mean(axis=ax))


def grid_to_config(grid_keys: list[str], param_combination: list[Any]) -> Configuration:
    """Convert grid parameters to a configuration object."""
    config = Configuration()
    for key, value in zip(grid_keys, param_combination):
        setattr(config, key, value)
    return config


def scenario_config(file: str, param_name: tuple[str], param_combinations: tuple[tuple]) -> Configuration:
    config = Configuration.from_file(file)

    for key, value in zip(param_name, param_combinations):
        setattr(config, key, value)

    # Add process noise
    if config.seed >= 0:
        np.random.seed(config.seed)  # For reproducibility
        random.seed(config.seed)

    config.populate_samples()

    single_benchmark(
        name=os.path.splitext(os.path.basename(file))[0],
        config=config,
    )


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
            with Stats() as stats:
                benchmark_pipeline(config=config)
                stats.collect_peak_rss_memory_usage()
                mlflow.log_metric("peak_rss_memory_usage_bytes", stats.peak_rss_memory_usage)
                mlflow.log_metric("C", stats.C)
                mlflow.log_metric("A_xn_wo_x0", stats.A_xn_wo_x0)
                mlflow.log_metric("A_xn_wo_xu", stats.A_xn_wo_xu)
                mlflow.log_metric("A_xn_wo_x", stats.A_xn_wo_x)
                mlflow.log_metric("min_x0", stats.min_x0)
                mlflow.log_metric("max_xn_wo_x0", stats.max_xn_wo_x0)
                mlflow.log_metric("max_xu", stats.max_xu)
                mlflow.log_metric("min_xn_wo_xu", stats.min_xn_wo_xu)
                mlflow.log_metric("max_x", stats.max_x)
                mlflow.log_metric("min_xn_wo_x", stats.min_xn_wo_x)
                mlflow.log_metric("min_d", stats.min_d)
                mlflow.log_metric("max_d_xn_wo_x", stats.max_d_xn_wo_x)
                mlflow.log_metric("num_constraints", stats.num_constraints)
                mlflow.log_metric("num_variables", stats.num_variables)
            status = mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.FINISHED)
        except Exception as ex:
            log.error(
                f"Error in benchmark {name} with configuration {config.to_safe_dict()}: {ex}\n{traceback.format_exc()}"
            )
            print(traceback.format_exc())
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


def tune(conf: Configuration, tffm: TruncatedFourierFeatureMap):
    import optuna

    if isinstance(conf.estimator, Estimator):
        return conf.estimator

    def sample(conf: Configuration) -> tuple["NMatrix", "NMatrix"]:
        if conf.system_dynamics is None:
            return conf.x_samples, conf.xp_samples
        X_samples = conf.X_bounds.sample(conf.num_samples)
        Xp_samples = conf.system_dynamics(X_samples)
        return X_samples, Xp_samples

    def objective(trial: optuna.Trial):
        sigma_l = np.array(
            [trial.suggest_float(f"sigma_l{i}", 1e-5, 1e5, log=True) for i in range(conf.X_bounds.dimension)]
        )
        estimator = conf.estimator(
            kernel=GaussianKernel(sigma_f=conf.sigma_f, sigma_l=sigma_l),
            regularization_constant=conf.lambda_,
        )
        training_x_samples, training_xp_samples = sample(conf)
        try:
            estimator.fit(training_x_samples, tffm(training_xp_samples))
        except Exception:
            return np.nan
        val_x_samples, val_xp_samples = sample(conf)
        return -estimator.score(val_x_samples, tffm(val_xp_samples))

    study = optuna.create_study()
    study.optimize(objective, n_trials=20, n_jobs=4)

    print("Number of finished trials: ", len(study.trials))
    best_params = study.best_params
    print("Best trial:", best_params)

    return conf.estimator(
        kernel=GaussianKernel(sigma_f=conf.sigma_f, sigma_l=np.array(tuple(best_params.values()))),
        regularization_constant=conf.lambda_,
    )


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
            not isinstance(config.feature_map, FeatureMap) or config.feature_map.num_frequencies == config.num_frequencies
        ), "num_frequencies and feature_map are mutually exclusive"
        assert (
            config.f_xp_samples is not None
            or config.feature_map is None
            or isinstance(config.feature_map, (FeatureMap, type))
        ), "f_xp_samples must be provided when feature_map is a callback"

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
        if not isinstance(estimator, ModelEstimator):
            log.info("Tuning hyperparameters...")
            estimator: "KernelRidgeRegressor[GaussianKernel]" = tune(config, feature_map)
            mlflow.log_param("fit.sigma_l", estimator.kernel.sigma_l.tolist())
        log.debug(f"Estimator pre-fit: {estimator}")
        estimator.fit(x=config.x_samples, y=config.f_xp_samples)  # Actual fitting of the regressor
        log.info(f"Estimator post-fit: {estimator}")

    with TimeLogger("evaluate"):
        if callable(feature_map) and not isinstance(feature_map, FeatureMap):
            feature_map = feature_map(estimator)  # Compute the feature map if it is a callable
        assert isinstance(feature_map, FeatureMap), "feature_map must return a FeatureMap instance"
        # for i, val in enumerate(rmse(estimator(config.x_samples), config.f_xp_samples)):
        #     mlflow.log_metric(f"f_xp_samples.rmse.{i}", val)
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
