import importlib
import json
from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import yaml
from jsonschema import validate
from jsonschema.exceptions import ValidationError

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


class ConfigAction(Action):
    """Custom action to handle configuration files."""

    def __call__(self, parser, namespace: CLIArgs, values: Path, option_string=None):
        """Parse the configuration file and update the namespace."""
        assert isinstance(values, Path), "Input must be a Path object"
        assert values.exists(), f"Configuration file does not exist: {values}"

        if values.suffix == ".py":  # No other actions, just return the path
            return setattr(namespace, self.dest, values)

        # We won't need to use the path later, just store an empty Path object
        setattr(namespace, self.dest, Path(""))

        # Load the configuration file and the JSON schema
        with open(values, "r", encoding="utf-8") as f:
            config = json.load(f) if values.suffix == ".json" else yaml.safe_load(f)
        with importlib.resources.open_text("pylucid", "cliargs_schema.json") as schema_file:
            schema = json.load(schema_file)

        # Validate the configuration against the schema
        try:
            validate(instance=config, schema=schema)
        except ValidationError as e:
            error_msg = f"Configuration file validation failed: {e.message}"
            raise raise_error(error_msg) from (e if namespace.verbose >= LOG_DEBUG else None)

        # Convert the dictionary to CLIArgs and update the namespace
        self.dict_to_cliargs(config, namespace)

    def dict_to_cliargs(self, config_dict: dict, args: CLIArgs) -> CLIArgs:
        """
        Convert a dictionary parsed from a YAML or JSON file to a CLIArgs object.

        Args:
            config_dict: Dictionary parsed from a configuration file

        Returns:
            CLIArgs object with the configuration values
        """
        # Process basic parameters
        setattr(args, "input", Path())
        setattr(args, "verbose", int(config_dict.get("verbose", args.verbose)))
        setattr(args, "seed", int(config_dict.get("seed", args.seed)))
        setattr(args, "gamma", float(config_dict.get("gamma", args.gamma)))
        setattr(args, "c_coefficient", float(config_dict.get("c_coefficient", args.c_coefficient)))
        # Note: JSON uses "lambda", not "lambda_"
        setattr(args, "lambda_", float(config_dict.get("lambda", args.lambda_)))
        setattr(args, "num_samples", int(config_dict.get("num_samples", args.num_samples)))
        setattr(args, "time_horizon", int(config_dict.get("time_horizon", args.time_horizon)))
        setattr(args, "num_frequencies", int(config_dict.get("num_frequencies", args.num_frequencies)))
        setattr(args, "oversample_factor", float(config_dict.get("oversample_factor", args.oversample_factor)))
        setattr(args, "num_oversample", int(config_dict.get("num_oversample", args.num_oversample)))
        setattr(args, "noise_scale", float(config_dict.get("noise_scale", args.noise_scale)))
        setattr(args, "plot", bool(config_dict.get("plot", args.plot)))
        setattr(args, "verify", bool(config_dict.get("verify", args.verify)))
        setattr(args, "problem_log_file", str(config_dict.get("problem_log_file", args.problem_log_file)))
        setattr(args, "iis_log_file", str(config_dict.get("iis_log_file", args.iis_log_file)))
        setattr(args, "sigma_f", float(config_dict.get("sigma_f", args.sigma_f)))

        # Handle sigma_l (can be single value or list)
        sigma_l = config_dict.get("sigma_l", args.sigma_l)
        if isinstance(sigma_l, list):
            args.sigma_l = np.array(sigma_l)
        elif isinstance(sigma_l, (int, float)):
            args.sigma_l = float(sigma_l)
        setattr(args, "sigma_l", sigma_l)

        # Process system dynamics
        system_dynamics = config_dict.get("system_dynamics", args.system_dynamics)
        if isinstance(system_dynamics, list):
            SystemDynamicsAction(option_strings=None, dest="system_dynamics")(None, args, system_dynamics)

        # Process sets
        set_parser = None
        for set_name in ("X_bounds", "X_init", "X_unsafe"):
            set_value = config_dict.get(set_name, getattr(args, set_name))
            if set_value is None or isinstance(set_value, Set):
                setattr(args, set_name, None)
            elif isinstance(set_value, str):
                set_parser = set_parser or SetParser()
                setattr(args, set_name, set_parser.parse(config_dict[set_name]))
            else:
                set_value = self.parse_set_from_dict(config_dict[set_name])
                setattr(args, set_name, set_value)

    def parse_set_from_dict(self, set_dict: dict):
        """Helper function to parse set objects from dictionary representation"""
        assert isinstance(set_dict, dict), "Input must be a dictionary representing a set"
        if "RectSet" in set_dict:
            rect_data = set_dict["RectSet"]
            return (
                RectSet(rect_data) if isinstance(rect_data, list) else RectSet(rect_data["lower"], rect_data["upper"])
            )
        if "MultiSet" in set_dict:
            multi_data = set_dict["MultiSet"]
            return MultiSet(*tuple(self.parse_set_from_dict(rect_item) for rect_item in multi_data))
        raise raise_error(f"Unsupported set type in dictionary: {set_dict}")


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
    parser.add_argument(
        "input",
        help="path to the configuration file. Can be a .py, .yaml or .json file. "
        "Command line parameter will override the values in the file. "
        "Python configuration files offer the most flexibility",
        nargs="?",
        action=ConfigAction,
        default=Path(""),
        type=type_valid_path,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=int,
        help=f"set verbosity level. "
        f"{log.LOG_NONE}: no output, "
        f"{log.LOG_CRITICAL}: critical, "
        f"{log.LOG_ERROR}: errors, "
        f"{log.LOG_WARN}: warning, "
        f"{log.LOG_INFO}: info, "
        f"{log.LOG_DEBUG}: debug, "
        f"{log.LOG_TRACE}: trace",
        choices=[
            log.LOG_NONE,
            log.LOG_CRITICAL,
            log.LOG_ERROR,
            log.LOG_WARN,
            log.LOG_INFO,
            log.LOG_DEBUG,
            log.LOG_TRACE,
        ],
        default=log.LOG_INFO,
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
