from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._pylucid import *
from ._pylucid import __version__

if TYPE_CHECKING:
    from typing import Callable

    from ._pyludic import NMatrix, NVector


@dataclass(frozen=True)
class ScenarioConfig:
    """Configuration used to launch the pipeline."""

    x_samples: "NMatrix"
    xp_samples: "NMatrix"
    X_bounds: "Set"
    X_init: "Set"
    X_unsafe: "Set"
    T: int = 5
    gamma: float = 1.0
    c_coefficient: float = 1.0
    f_xp_samples: "NMatrix | Callable[[NMatrix], NMatrix] | None" = None
    f_det: "Callable[[NMatrix], NMatrix] | None" = None
    estimator: "Estimator | None" = None
    num_freq_per_dim: int = -1
    oversample_factor: float = 2.0
    num_oversample: int = -1
    feature_map: "FeatureMap | type[FeatureMap] | Callable[[Estimator], FeatureMap] | None" = None
    sigma_f: float = 1.0
    noise_scale: float = 0.01
    verify: bool = True
    plot: bool = True
    problem_log_file: str = ""
    iis_log_file: str = ""

    def keys(self) -> "list[str]":
        """Returns a list of keys for the configuration attributes."""
        return [
            "x_samples",
            "xp_samples",
            "X_bounds",
            "X_init",
            "X_unsafe",
            "T",
            "gamma",
            "c_coefficient",
            "f_xp_samples",
            "f_det",
            "estimator",
            "num_freq_per_dim",
            "oversample_factor",
            "num_oversample",
            "feature_map",
            "sigma_f",
            "noise_scale",
            "verify",
            "plot",
            "problem_log_file",
            "iis_log_file",
        ]

    def __getitem__(self, key) -> "NMatrix | Callable[[NMatrix], NMatrix] | Set | int | float | str | None":
        return getattr(self, key)


class CLIArgs(Namespace):
    """Command line arguments for the CLI interface."""

    verbose: int
    input: Path
    seed: int
    gamma: float
    c_coefficient: float
    lambda_: float  # Use 'lambda_' to avoid conflict with the Python keyword 'lambda'
    sigma_f: float
    sigma_l: "NVector | float"  # Can be a single float or an array of floats
    num_samples: int
    time_horizon: int
    noise_scale: float
    plot: bool
    verify: bool
    problem_log_file: str
    iis_log_file: str
    num_frequencies: int  # Default number of frequencies per dimension for the Fourier feature map
    oversample_factor: float
    num_oversample: int


def cli_scenario_config(args: CLIArgs) -> "ScenarioConfig":
    # If a seed is provided, set the random seed for reproducibility
    if args.seed >= 0:
        np.random.seed(args.seed)

    f_det = lambda x: np.array([x[:, 1], -x[:, 0] - x[:, 1] + 1 / 3 * x[:, 0] ** 3]).T  # lambda x: x
    f = lambda x: f_det(x) + (np.random.standard_normal())

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
    x_samples = X_bounds.sample(args.num_samples)
    xp_samples = f_det(x_samples)

    # Initial estimator hyperparameters. Can be tuned later

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
        num_frequencies=args.num_frequencies,
        sigma_l=args.sigma_l,
        sigma_f=args.sigma_f,
        x_limits=X_bounds,
    )
    estimator = ModelEstimator(f=lambda x: feature_map(f_det(x)))  # Use the custom model estimator

    return ScenarioConfig(
        x_samples=x_samples,
        xp_samples=xp_samples,
        X_bounds=X_bounds,
        X_init=X_init,
        X_unsafe=X_unsafe,
        T=args.time_horizon,
        gamma=args.gamma,
        f_det=f_det,  # The deterministic part of the system dynamics
        # num_freq_per_dim=num_freq_per_dim,  # Number of frequencies per dimension for the Fourier feature map
        estimator=estimator,  # The estimator used to model the system dynamics
        sigma_f=args.sigma_f,
        problem_log_file=args.problem_log_file,  # The lp file containing the optimization problem
        iis_log_file=args.iis_log_file,  # The ilp file containing the irreducible infeasible set (IIS) if the problem is infeasible
        feature_map=feature_map,  # The feature map used to transform the state space
        plot=args.plot,  # Whether to plot the barrier certificate
        verify=args.verify,  # Whether to verify the barrier certificate using an SMT solver
    )


def arg_parser() -> "ArgumentParser":
    def valid_path(path_str: str) -> Path:
        if not path_str:
            return Path()  # Allow empty input for default behavior
        path = Path(path_str)
        if not path.exists():
            raise raise_error(f"Path does not exist: {path_str}")
        if not path.is_file() or not path.suffix == ".py":
            raise raise_error(f"Path must be a Python file: {path_str}")
        return path

    parser = ArgumentParser(prog="pylucid", description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("input", help="path to the input file", nargs="?", default="", type=valid_path)
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
        "-c",
        "--c_coefficient",
        type=float,
        default=1.0,
        help="coefficient to make the barrier certificate more (> 1) or less (< 1) conservative",
    )
    parser.add_argument(
        "-l",
        "--lambda",
        dest="lambda_",
        type=float,
        default=1.0,
        help="regularization constant for the estimator",
    )
    parser.add_argument(
        "-N",
        "--num_samples",
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
    parser.add_argument(
        "-T",
        "--time_horizon",
        type=int,
        default=10,
        help="time horizon for the scenario",
    )
    parser.add_argument(
        "-f",
        "--num_frequencies",
        type=int,
        default=4,
        help="number of frequencies per dimension for the Fourier feature map",
    )
    parser.add_argument(
        "--oversample_factor",
        type=float,
        default=2.0,
        help="factor by which to oversample the frequency space. If `num_oversample` is set, it takes precedence",
    )
    parser.add_argument(
        "--num_oversample",
        type=int,
        default=-1,
        help="number of samples to use for the frequency space. If negative, it is computed based on the oversample factor",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.01,
        help="scale of the noise added to the input samples. If 0, no noise is added.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="plot the barrier certificate, if available. Requires matplotlib",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify the barrier certificate using an SMT solver, if available. Requires dReal",
    )
    parser.add_argument(
        "--problem_log_file",
        type=str,
        default="",
        help="file to save the optimization problem in LP format. If empty, the problem will not be saved.",
    )
    parser.add_argument(
        "--iis_log_file",
        type=str,
        default="",
        help="file to save the irreducible infeasible set (IIS) in ILP format. If empty, the IIS will not be saved.",
    )

    return parser
