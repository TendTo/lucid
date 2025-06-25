from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ._pylucid import *
from ._pylucid import __version__
from .parser import SetParser, SympyParser
from .util import raise_error

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
            "verify",
            "plot",
            "problem_log_file",
            "iis_log_file",
        ]

    def __getitem__(self, key) -> "NMatrix | Callable[[NMatrix], NMatrix] | Set | int | float | str | None":
        return getattr(self, key)


class CLIArgs(Namespace):
    """Command line arguments for the CLI interface."""

    system_dynamics: "Callable[[NMatrix], NMatrix] | None"
    X_bounds: "Set | None"
    X_init: "Set | None"
    X_unsafe: "Set | None"
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


class FloatOrNVectorAction(Action):
    def __init__(self, **kwargs):
        super().__init__(nargs="+", **kwargs)

    def __call__(self, parser, namespace, values: "list[float]", option_string=None):
        setattr(namespace, self.dest, np.array(values) if len(values) > 1 else values[0])


class SystemDynamicsAction(Action):
    def __init__(self, **kwargs):
        super().__init__(nargs="+", **kwargs)

    def __call__(self, parser, namespace, values: "list[str]", option_string=None):
        sym_parser = SympyParser()
        functions = sym_parser.parse_to_lambda(values)

        def system_dynamics_func(x: "NMatrix") -> "NMatrix":
            """Dynamic function that takes a state vector and returns the next state."""
            assert x.ndim == 2, "Input must be a 2D array with shape (n_samples, n_features)"
            cols = {f"x{i + 1}": x[:, i] for i in range(x.shape[1])}
            return np.column_stack(tuple(f(**cols) for f in functions))

        setattr(namespace, self.dest, system_dynamics_func if functions else None)


def type_valid_path(path_str: str) -> Path:
    if not path_str:
        return Path()  # Allow empty input for default behavior
    path = Path(path_str)
    if not path.exists():
        raise raise_error(f"Path does not exist: {path_str}")
    supported_types = (".py", ".yaml", ".json", ".yml")
    if not path.is_file() or path.suffix not in supported_types:
        raise raise_error(f"Supported file types are {supported_types}. Invalid file: {path_str}")
    return path


def type_set(set_str: str) -> "Set":
    """Convert a string representation of a function into a callable."""
    return SetParser().parse(set_str)


def arg_parser() -> "ArgumentParser":
    parser = ArgumentParser(prog="pylucid", description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("input", help="path to the input file", nargs="?", default="", type=type_valid_path)
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
        action=FloatOrNVectorAction,
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
    parser.add_argument(
        "--system_dynamics",
        type=str,
        default=None,
        action=SystemDynamicsAction,
        help="system dynamics function as a string. "
        "Specify a function for each dimension of the output space. "
        "Variables `x1`, `x2`, ..., `xn` stand for the n-dimensional input state space components. "
        "All components of the input state space must be present in the function. "
        "For example, `--system_dynamics 'x1**2 + x2 / 2' '2 * x1 + sin(-x2)' 'cos(x1)'` "
        "will produce a function that takes a 2D input (x1, x2) and returns a 3D output (y1, y2, y3).",
    )
    parser.add_argument(
        "--X_bounds",
        type=type_set,
        default=None,
        help="state space X bounds as a string. For example, `--X_bounds 'RectSet([-3, -2], [2.5, 1])'`",
    )
    parser.add_argument(
        "--X_init",
        type=type_set,
        default=None,
        help="initial set X_0 as a string. "
        "For example, `--X_init 'MultiSet(RectSet([1, -0.5], [2, 0.5]), RectSet([-1.8, -0.1], [-1.2, 0.1]))'`",
    )
    parser.add_argument(
        "--X_unsafe",
        type=type_set,
        default=None,
        help="unsafe set X_U as a string. "
        "For example, `--X_unsafe 'MultiSet(RectSet([0.4, 0.1], [0.6, 0.5]), RectSet([0.4, 0.1], [0.8, 0.3]))'`",
    )

    return parser
