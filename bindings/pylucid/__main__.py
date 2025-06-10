from pylucid import *
from pylucid import __version__
from typing import TYPE_CHECKING
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from pathlib import Path
import importlib
import time
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


def raise_error(message: str, error_type: type = ValueError):
    """Raise an error with the given message."""
    log_error(message)
    return error_type(message)


class CLIArgs(Namespace):
    verbose: int
    input: Path
    seed: int
    gamma: float
    lambda_: float  # Use 'lambda_' to avoid conflict with the Python keyword 'lambda'
    sigma_f: float
    sigma_l: "np.typing.NDArray[np.float64] | float"  # Can be a single float or an array of floats
    N: int


def cli_scenario_config(args: CLIArgs) -> "ScenarioConfig":
    # If a seed is provided, set the random seed for reproducibility
    if args.seed >= 0:
        np.random.seed(args.seed)

    f_det = lambda x: np.array([x[:, 1], -x[:, 0] - x[:, 1] + 1 / 3 * x[:, 0] ** 3]).T  # lambda x: x
    f = lambda x: f_det(x) + (np.random.standard_normal())

    gamma = 18.312
    T = 10  # Time horizon

    X_bounds = RectSet((-3, -2), (2.5, 1), seed=args.seed)  # State space X
    # Initial set X_0
    X_init = MultiSet(
        RectSet((1, -0.5), (2, 0.5)),
        RectSet((-1.8, -0.1), (-1.2, 0.1)),
        RectSet((-1.4, -0.5), (-1.2, 0.1)),
    )
    # Unsafe set X_U
    X_unsafe = MultiSet(RectSet((0.4, 0.1), (0.6, 0.5)), RectSet((0.4, 0.1), (0.8, 0.3)))

    # Parameters and inputs
    x_samples = X_bounds.sample(args.N)
    xp_samples = f_det(x_samples)

    # Initial estimator hyperparameters. Can be tuned later
    sigma_f = 15.0
    sigma_l = np.array([1, 1.0])

    num_freq_per_dim = 4  # Number of frequencies per dimension. Includes the zero frequency.

    # De-comment the tuner you want to use or leave it empty to avoid tuning.
    tuner = {
        # "tuner": LbfgsTuner(bounds=((1e-5, 1e5), (1e-5, 1e5)), parameters=LbgsParameters(min_step=0, linesearch=5))
        # "tuner": MedianHeuristicTuner(),
        # "tuner": GridSearchTuner(
        #     ParameterValues(
        #         Parameter.SIGMA_L, [np.full(2, v) for v in np.linspace(0.1, 15.0, num=10, endpoint=True, dtype=float)]
        #     ),
        #     ParameterValues(Parameter.SIGMA_F, np.linspace(0.1, 15.0, num=10, endpoint=True, dtype=float)),
        #     ParameterValues(Parameter.REGULARIZATION_CONSTANT, np.logspace(-6, -1, num=10)),
        # ),
    }
    # estimator = KernelRidgeRegressor(
    #     kernel=GaussianKernel(sigma_f=sigma_f, sigma_l=sigma_l),
    #     regularization_constant=regularization_constant,
    # )
    feature_map = ConstantTruncatedFourierFeatureMap(
        num_frequencies=num_freq_per_dim,
        sigma_l=sigma_l,
        sigma_f=sigma_f,
        x_limits=X_bounds,
    )
    print(f"Feature map: {feature_map(f_det(x_samples))}")
    estimator = ModelEstimator(f=lambda x: feature_map(f_det(x)))  # Use the custom model estimator

    return ScenarioConfig(
        x_samples=x_samples,
        xp_samples=xp_samples,
        X_bounds=X_bounds,
        X_init=X_init,
        X_unsafe=X_unsafe,
        T=T,
        gamma=gamma,
        f_det=f_det,  # The deterministic part of the system dynamics
        # num_freq_per_dim=num_freq_per_dim,  # Number of frequencies per dimension for the Fourier feature map
        estimator=estimator,  # The estimator used to model the system dynamics
        sigma_f=sigma_f,
        problem_log_file="problem.lp",  # The lp file containing the optimization problem
        iis_log_file="iis.ilp",  # The ilp file containing the irreducible infeasible set (IIS) if the problem is infeasible
        feature_map=feature_map,  # The feature map used to transform the state space
    )


def arg_parser() -> "ArgumentParser":
    def valid_path(path_str: str) -> Path:
        path = Path(path_str)
        if not path.exists():
            raise raise_error(f"Path does not exist: {path_str}")
        if not path.is_file() or not path.suffix == ".py":
            raise raise_error(f"Path must be a Python file: {path_str}")
        return path

    parser = ArgumentParser(prog="pylucid", description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("input", help="path to the input file", type=valid_path)
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        help=f"set verbosity level. {LOG_NONE}: no output, {LOG_CRITICAL}: critical, {LOG_ERROR}: errors, {LOG_WARN}: warning, {LOG_INFO}: info, {LOG_DEBUG}: debug, {LOG_TRACE}: trace",
        choices=[LOG_NONE, LOG_CRITICAL, LOG_ERROR, LOG_WARN, LOG_INFO, LOG_DEBUG, LOG_TRACE],
        default=LOG_INFO,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=-1,
        help="random seed for reproducibility. Set to < 0 to disable seeding",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=1.0,
        help="discount factor for future rewards",
    )
    parser.add_argument(
        "-l",
        "--lambda",
        type=float,
        default=1.0,
        help="regularization constant for the estimator",
    )
    parser.add_argument(
        "-N",
        "--num-samples",
        type=int,
        default=1000,
        help="number of samples to use to train the estimator",
    )
    parser.add_argument(
        "--sigma_f",
        type=float,
        default=15.0,
        help="hyperparameter for the feature map, sigma_f",
    )
    parser.add_argument(
        "--sigma_l",
        default=[1.0],
        type=float,
        nargs="+",
        help="hyperparameter for the feature map, sigma_l",
    )

    return parser


def main(argv: "Sequence[str] | None" = None):
    args = arg_parser().parse_args(argv, namespace=CLIArgs())
    args.sigma_l = np.array(args.sigma_l) if len(args.sigma_l) > 1 else args.sigma_l[0]
    # Set verbosity based on the command line argument
    set_verbosity(args.verbose)
    # Import the input file as a module
    mod = importlib.import_module(".".join(args.input.parts).removesuffix(".py"))
    # Check if the module has a 'scenario_config' function
    if not hasattr(mod, "scenario_config"):
        raise raise_error("The configuration file must contain a function called 'scenario_config'")
    # Validate the configuration returned by the 'scenario_config' function
    config: ScenarioConfig = mod.scenario_config()
    if not isinstance(config, (ScenarioConfig, dict)):
        raise raise_error("The 'scenario_config' function must return an instance of 'ScenarioConfig' or a dict")

    # If all the checks pass, run the scenario
    log_info(f"Running scenario (LUCID version: {__version__})")
    from pylucid.pipeline import pipeline

    start = time.time()
    pipeline(**config)
    end = time.time()
    log_info(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    main()
