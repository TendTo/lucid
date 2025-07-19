import importlib
import json
import sys
from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import scipy.io
import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from ._pylucid import *
from ._pylucid import __version__
from .parser import SetParser, SympyParser
from .util import assert_or_raise, raise_error

if TYPE_CHECKING:
    from typing import Any, Callable

    from ._pylucid import NMatrix, NVector


@dataclass
class Configuration(Namespace):
    """Configuration determining the scenario.

    Attributes:
        verbose: Verbosity level for logging
        seed: Random seed for reproducibility. If < 0, no seeding is done
        input: Path to the configuration file. Can be a .py, .yaml or .json file.

        system_dynamics: Deterministic function that maps the state variable x to the next state variable x'
        X_bounds: Set representing the bounds of the state space
        X_init: Set representing the initial states
        X_unsafe: Set representing the unsafe states

        x_samples: Input samples for the state variable x
        xp_samples: Input samples for the next state variable x'
        f_xp_samples: Precomputed samples of the next state variable x' or a function that computes them

        gamma: Discount or scaling factor for the optimization
        c_coefficient: coefficient that can be used to make the optimization more (> 1) or less (< 1) conservative
        lambda_: Regularization constant for the estimator
        sigma_f: Estimated mean parameter for the feature map
        sigma_l: Signal variance parameter for the feature map, can be a single float or an array of floats
        num_samples: Number of samples to use for training the estimator
        time_horizon: Time horizon for the scenario
        noise_scale: Scale of the noise added to the input samples. If 0, no noise is added.
        plot: Whether to plot the solution using matplotlib
        verify: Whether to verify the barrier certificate using dReal
        problem_log_file: File to save the optimization problem in LP format. If empty, the problem will not be saved.
        iis_log_file: File to save the irreducible infeasible set (IIS) in ILP format. If empty, the IIS will not be saved

        num_frequencies: Number of frequencies per dimension for the feature map
        oversample_factor: Factor by which to oversample the frequency space
        num_oversample: Number of samples to use for the frequency space. If negative, it is computed based on the oversample_factor

        estimator: Estimator class to use for regression
        kernel: Kernel class to use for the estimator
        feature_map: Feature map class to use for transformation or a callable that returns a feature map
        optimiser: Optimiser class to use for the optimization
        plot: Whether to plot the solution using matplotlib
    """

    verbose: int = log.LOG_INFO  # Default verbosity level
    seed: int = -1  # Default seed, -1 means no seeding
    input: Path = field(default_factory=Path)  # Default to empty Path, can be set by ConfigAction

    system_dynamics: "Callable[[NMatrix], NMatrix] | None" = None
    X_bounds: "Set | None" = None
    X_init: "Set | None" = None
    X_unsafe: "Set | None" = None

    x_samples: "NMatrix" = field(default_factory=lambda: np.empty((0, 0), dtype=np.float64))
    xp_samples: "NMatrix | Callable[[NMatrix], NMatrix]" = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float64)
    )
    f_xp_samples: "NMatrix | Callable[[NMatrix], NMatrix] | None" = None

    gamma: float = 1.0
    c_coefficient: float = 1.0
    lambda_: float = 1e-6  # Use 'lambda_' to avoid conflict with the Python keyword 'lambda'
    sigma_f: float = 1.0  # Hyperparameter for the feature map, sigma_f
    sigma_l: "NVector | float" = 1.0  # Can be a single float or an array of floats
    num_samples: int = 1000  # Default number of samples
    time_horizon: int = 5  # Default time horizon
    noise_scale: float = 0.01  # Default noise scale
    plot: bool = False  # Default plot flag
    verify: bool = False  # Default verify flag
    problem_log_file: str = ""
    iis_log_file: str = ""

    num_frequencies: int = 10  # Default number of frequencies per dimension for the Fourier feature map
    oversample_factor: float = 2.0
    num_oversample: int = (
        -1
    )  # Default number of oversamples, -1 means it will be computed based on the oversample factor
    estimator: "type[Estimator]" = KernelRidgeRegressor
    kernel: "type[Kernel]" = GaussianKernel
    feature_map: "type[FeatureMap] | FeatureMap | Callable[[Estimator], FeatureMap]" = LinearTruncatedFourierFeatureMap
    optimiser: "type[Optimiser]" = GurobiOptimiser if GUROBI_BUILD else AlglibOptimiser
    tuner: "Tuner | None" = None  # Tuner for the estimator, if any

    def to_safe_dict(self) -> dict:
        config_dict = asdict(self)  # Convert the Configuration object to a dictionary
        for k, v in config_dict.items():
            if isinstance(v, np.ndarray):
                config_dict[k] = v.tolist()
            if isinstance(v, type):
                config_dict[k] = v.__name__
            if isinstance(v, Path):
                config_dict[k] = str(v)
        return config_dict

    def to_yaml(self, path: "str | Path | None" = None) -> str:
        """Convert the configuration to a YAML string or save it to a file."""
        import yaml

        config_dict = self.to_safe_dict()
        yaml_str = yaml.safe_dump(config_dict, default_flow_style=False, sort_keys=False)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(yaml_str)
        return yaml_str


class ConfigAction(Action):
    """Custom action to handle configuration files."""

    def __call__(self, parser, namespace: Configuration, values: Path, option_string=None):
        """Parse the configuration file and update the namespace."""
        assert isinstance(values, Path), "Input must be a Path object"
        assert_or_raise(values.exists(), f"Configuration file does not exist: {values}")

        setattr(namespace, self.dest, values)
        if values.suffix == ".py" or values == Path():
            # We don't need to load a config file, just set the input path as provided
            return

        suffixes = (".py", ".yaml", ".json", ".yml")
        assert_or_raise(
            values.suffix in suffixes, f"Unsupported file type: {values.suffix}. Supported types are {suffixes}"
        )

        # We won't need to use the path later, just store an empty Path object

        # Load the configuration file and the JSON schema
        with open(values, "r", encoding="utf-8") as f:
            config = json.load(f) if values.suffix == ".json" else yaml.safe_load(f)
        if not isinstance(config, dict):
            raise raise_error(f"Configuration file must contain a dictionary, got {type(config)} instead")
        # Validate the configuration dictionary against the schema
        self.validate(config, verbosity=namespace.verbose)

        # Convert the dictionary to configuration and update the namespace
        self.dict_to_configuration(config, namespace)

    def validate(self, config_dict: dict, verbosity: int = log.LOG_INFO):
        """Validate the configuration dictionary against the schema."""
        if sys.version_info < (3, 9):
            with importlib.resources.open_text("pylucid", "configuration_schema.json", encoding="utf-8") as schema_file:
                schema = json.load(schema_file)
        else:
            with importlib.resources.files("pylucid").joinpath("configuration_schema.json").open(
                "r", encoding="utf-8"
            ) as schema_file:
                schema = json.load(schema_file)

        try:
            Draft202012Validator(schema).validate(instance=config_dict)
        except ValidationError as e:
            raise raise_error(f"{e.message}", ValidationError) from (e if verbosity >= log.LOG_DEBUG else None)

    def dict_to_configuration(self, config_dict: dict, args: Configuration) -> Configuration:
        """
        Convert a dictionary parsed from a YAML or JSON file to a Configuration object.

        Args:
            config_dict: Dictionary parsed from a configuration file

        Returns:
            Configuration object with the configuration values
        """
        # Process basic parameters
        args.input = Path()
        args.verbose = int(config_dict.get("verbose", args.verbose))
        args.seed = int(config_dict.get("seed", args.seed))
        args.gamma = float(config_dict.get("gamma", args.gamma))
        args.c_coefficient = float(config_dict.get("c_coefficient", args.c_coefficient))
        # Note: JSON uses "lambda", not "lambda_"
        args.lambda_ = float(config_dict.get("lambda", args.lambda_))
        args.num_samples = int(config_dict.get("num_samples", args.num_samples))
        args.time_horizon = int(config_dict.get("time_horizon", args.time_horizon))
        args.num_frequencies = int(config_dict.get("num_frequencies", args.num_frequencies))
        args.oversample_factor = float(config_dict.get("oversample_factor", args.oversample_factor))
        args.num_oversample = int(config_dict.get("num_oversample", args.num_oversample))
        args.noise_scale = float(config_dict.get("noise_scale", args.noise_scale))
        args.plot = bool(config_dict.get("plot", args.plot))
        args.verify = bool(config_dict.get("verify", args.verify))
        args.problem_log_file = str(config_dict.get("problem_log_file", args.problem_log_file))
        args.iis_log_file = str(config_dict.get("iis_log_file", args.iis_log_file))
        args.sigma_f = float(config_dict.get("sigma_f", args.sigma_f))

        EstimatorAction(option_strings=None, dest="estimator")(None, args, config_dict.get("estimator", args.estimator))
        KernelAction(option_strings=None, dest="kernel")(None, args, config_dict.get("kernel", args.kernel))
        FeatureMapAction(option_strings=None, dest="feature_map")(
            None, args, config_dict.get("feature_map", args.feature_map)
        )
        OptimiserAction(option_strings=None, dest="optimiser")(None, args, config_dict.get("optimiser", args.optimiser))
        NMatrixAction(option_strings=None, dest="x_samples")(None, args, config_dict.get("x_samples", args.x_samples))
        NMatrixAction(option_strings=None, dest="xp_samples")(
            None, args, config_dict.get("xp_samples", config_dict.get("f_xp_samples", args.xp_samples))
        )
        FloatOrNVectorAction(option_strings=None, dest="sigma_l")(None, args, config_dict.get("sigma_l", args.sigma_l))

        # Process system dynamics
        system_dynamics = config_dict.get("system_dynamics", args.system_dynamics)
        if isinstance(system_dynamics, list) and len(system_dynamics) > 0:
            SystemDynamicsAction(option_strings=None, dest="system_dynamics")(None, args, system_dynamics)

        # Process sets
        set_parser = None
        for set_name in ("X_bounds", "X_init", "X_unsafe"):
            set_value = config_dict.get(set_name, getattr(args, set_name))
            if set_value is None or isinstance(set_value, Set):
                setattr(args, set_name, set_value)
            elif isinstance(set_value, str):
                set_parser = set_parser or SetParser()
                setattr(args, set_name, set_parser.parse(config_dict[set_name]))
            else:
                set_value = self.parse_set_from_config(config_dict[set_name])
                setattr(args, set_name, set_value)

    def parse_set_from_config(self, set_config: "dict | list | str") -> "Set":
        """Helper function to parse set objects from dictionary/list representation"""
        if isinstance(set_config, str):
            set_parser = SetParser()
            return set_parser.parse(set_config)
        if not isinstance(set_config, dict):
            if len(set_config) == 0:
                raise raise_error("Set configuration cannot be empty")
            if len(set_config) == 1 and isinstance(set_config[0], dict):
                return self.parse_set_from_config(set_config[0])
            return MultiSet(*tuple(self.parse_set_from_config(rect_item) for rect_item in set_config))
        assert isinstance(set_config, dict), "Set configuration must be a dictionary or a list of dictionaries"
        if "RectSet" in set_config:
            rect_data = set_config["RectSet"]
            return (
                RectSet(rect_data["lower"], rect_data["upper"]) if isinstance(rect_data, dict) else RectSet(rect_data)
            )
        if "SphereSet" in set_config:
            sphere_data = set_config["SphereSet"]
            return SphereSet(sphere_data["center"], sphere_data["radius"])
        raise raise_error(f"Unsupported set type in dictionary: {set_config}")


class FloatOrNVectorAction(Action):
    def __init__(self, **kwargs):
        super().__init__(nargs="+", **kwargs)

    def __call__(self, parser, namespace, values: "float | list[float]", option_string=None):
        if isinstance(values, (float, int)):
            values = [values]
        setattr(namespace, self.dest, np.array(values, dtype=np.float64) if len(values) > 1 else float(values[0]))


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
            f_cols = [f(**cols) for f in functions]
            rows = max(len(f_col) for f_col in f_cols if isinstance(f_col, np.ndarray))
            # Broadcast scalar values to match the number of rows of the output
            for i, f_col in enumerate(f_cols):
                if np.isscalar(f_col):
                    f_cols[i] = np.full((rows,), f_col, dtype=np.float64)
            return np.column_stack(f_cols)

        setattr(namespace, self.dest, system_dynamics_func if functions else None)


def type_valid_path(path_str: str) -> Path:
    if not path_str:
        return Path()  # Allow empty input for default behavior
    path = Path(path_str)
    if not path.exists():
        raise raise_error(f"Path does not exist: {path_str}")
    return path


def type_set(set_str: str) -> "Set":
    """Convert a string representation of a function into a callable."""
    return SetParser().parse(set_str)


class EstimatorAction(Action):
    def __call__(self, parser, namespace, values: "str | type[Estimator]", option_string=None):
        if isinstance(values, type) and issubclass(values, Estimator):
            return setattr(namespace, self.dest, values)
        if values == "KernelRidgeRegressor":
            return setattr(namespace, self.dest, KernelRidgeRegressor)
        raise raise_error(f"Unsupported estimator type: {values}")


class KernelAction(Action):
    def __call__(self, parser, namespace, values: "str | type[Kernel]", option_string=None):
        if isinstance(values, type) and issubclass(values, Kernel):
            return setattr(namespace, self.dest, values)
        if values == "GaussianKernel":
            return setattr(namespace, self.dest, GaussianKernel)
        raise raise_error(f"Unsupported kernel type: {values}")


class FeatureMapAction(Action):
    def __call__(self, parser, namespace, values: "str | type[FeatureMap]", option_string=None):
        if isinstance(values, type) and issubclass(values, FeatureMap):
            return setattr(namespace, self.dest, values)
        if values == "LogTruncatedFourierFeatureMap":
            return setattr(namespace, self.dest, LogTruncatedFourierFeatureMap)
        if values == "ConstantTruncatedFourierFeatureMap":
            return setattr(namespace, self.dest, ConstantTruncatedFourierFeatureMap)
        if values == "LinearTruncatedFourierFeatureMap":
            return setattr(namespace, self.dest, LinearTruncatedFourierFeatureMap)
        raise raise_error(f"Unsupported feature map type: {values}")


class OptimiserAction(Action):
    def __call__(self, parser, namespace, values: "str | type[Optimiser]", option_string=None):
        if isinstance(values, type) and issubclass(values, Optimiser):
            return setattr(namespace, self.dest, values)
        if values == "GurobiOptimiser":
            return setattr(namespace, self.dest, GurobiOptimiser)
        if values == "AlglibOptimiser":
            return setattr(namespace, self.dest, AlglibOptimiser)
        if values == "HighsOptimiser":
            return setattr(namespace, self.dest, HighsOptimiser)
        raise raise_error(f"Unsupported optimiser type: {values}")


class NMatrixAction(Action):
    def __call__(self, parser, namespace, values: "str | type[NMatrix]", option_string=None):
        if values is None:
            return setattr(namespace, self.dest, np.empty((0, 0), dtype=np.float64))
        if isinstance(values, np.ndarray):
            return setattr(namespace, self.dest, values)
        if isinstance(values, list):
            return setattr(namespace, self.dest, np.array(values, dtype=np.float64))
        suffixes = (".npy", ".npz", ".csv", ".mat")
        path_to_file = Path(values)
        if path_to_file.suffix in suffixes:
            assert_or_raise(path_to_file.exists(), f"File does not exist: {values}")
            if path_to_file.suffix in (".npy", ".npz"):
                data = np.load(path_to_file, allow_pickle=True)
                if isinstance(data, np.lib.npyio.NpzFile):
                    # If it's a .npz file, we need to extract the first array
                    data = next(iter(data.values()))
            elif path_to_file.suffix == ".csv":
                with open(path_to_file, "rb") as f:
                    # Load CSV data, assuming it is a 2D array
                    lines = f.readlines()
                data = np.genfromtxt(lines, delimiter=",", dtype=np.float64).reshape(len(lines), -1)
                data = data[~np.isnan(data).any(axis=1)]
            elif path_to_file.suffix == ".mat":
                mat_data: "dict[str, Any]" = scipy.io.loadmat(path_to_file)
                for k, v in mat_data.items():
                    if k.startswith("__"):
                        continue
                    data = v
        else:
            try:
                data = np.array(json.loads(values), dtype=np.float64)
            except json.JSONDecodeError:
                raise raise_error(f"Invalid JSON string: {values}")
        assert_or_raise(isinstance(data, np.ndarray), f"Expected a numpy array, got {type(data)}")
        assert_or_raise(len(data) > 0, f"CSV file is empty: {values}")
        assert_or_raise(data.ndim == 2, f"Data must be a 2D array, got {data.ndim}D array instead")
        return setattr(namespace, self.dest, data)


class MultiNMatrixAction(Action):
    def __call__(self, parser, namespace: Configuration, values: "str", option_string=None):
        assert_or_raise(isinstance(values, str), "Input must be a string representing a file path and parsing info")
        if ":" in values:
            path_to_file, info = values.split(":")
            path_to_file = Path(path_to_file)
        else:
            path_to_file, info = Path(values), ""
        assert_or_raise(path_to_file.exists(), f"File does not exist: {values}")
        if path_to_file.suffix == ".npz":
            raise NotImplementedError("MultiNMatrixAction does not support .npz files with multiple arrays. ")
        elif path_to_file.suffix == ".csv":
            cols = int(info) if info.isdigit() else (namespace.X_bounds.dimension if namespace.X_bounds else 0)
            assert_or_raise(cols > 0, f"Invalid number of columns specified: {cols}")
            with open(path_to_file, "rb") as f:
                # Load CSV data, assuming it is a 2D array
                lines = f.readlines()
            data = np.genfromtxt(lines, delimiter=",", dtype=np.float64).reshape(len(lines), -1)
            data = data[~np.isnan(data).any(axis=1)]
            x_samples, xp_samples = data[:, :cols], data[:, cols:]
        elif path_to_file.suffix == ".mat":
            if "," in info:
                x_key, xp_key = info.split(",")
            else:
                x_key, xp_key = "x_samples", "xp_samples"
            mat_data: "dict[str, Any]" = scipy.io.loadmat(path_to_file)
            x_samples, xp_samples = mat_data.get(x_key), mat_data.get(xp_key)
        for data in (x_samples, xp_samples):
            assert_or_raise(isinstance(data, np.ndarray), f"Expected a numpy array, got {type(data)}")
            assert_or_raise(len(data) > 0, f"CSV file is empty: {values}")
            assert_or_raise(data.ndim == 2, f"Data must be a 2D array, got {data.ndim}D array instead")
        namespace.x_samples = x_samples
        namespace.xp_samples = xp_samples


def arg_parser() -> "ArgumentParser":
    config = Configuration()
    parser = ArgumentParser(prog="pylucid", description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "input",
        help="path to the configuration file. Can be a .py, .yaml or .json file. "
        "Command line parameter will override the values in the file. "
        "Python configuration files offer the most flexibility",
        nargs="?",
        action=ConfigAction,
        default=Path(),
        type=type_valid_path,
    )
    parser.add_argument(
        "--samples",
        help="path to the samples file followed by additional information on how to parse it. "
        "Can be a .npz, .csv or .mat file. The additional information follows a colon (:). "
        "For .mat files, you must specify the variable names to extract, e.g., '/path/to/samples.mat:X,XP'. "
        "For .csv files, you must specify the input dimension, e.g., '/path/to/samples.csv:3'. ",
        nargs="?",
        action=MultiNMatrixAction,
        type=str,
    )
    parser.add_argument(
        "-i",
        "--x-samples",
        help="json-like 2D array or path to the input samples file. "
        "Can be a .npy, .npz or .csv file. "
        "If not provided, the input samples will be collected from the state space bounds",
        action=NMatrixAction,
        type=str,
    )
    parser.add_argument(
        "-o",
        "--xp-samples",
        help="json-like 2D array or path to the next state samples file. "
        "Can be a .npy, .npz or .csv file. "
        "If not provided, the next state samples will be computed from the x_samples using the system dynamics function",
        action=NMatrixAction,
        type=str,
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
        default=config.verbose,
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=config.seed,
        help="random seed for reproducibility. Set to < 0 to disable seeding",
    )
    parser.add_argument(
        "-g",
        "--gamma",
        type=float,
        default=config.gamma,
        help="discount factor for future rewards",
    )
    parser.add_argument(
        "-c",
        "--c_coefficient",
        type=float,
        default=config.c_coefficient,
        help="coefficient to make the barrier certificate more (> 1) or less (< 1) conservative",
    )
    parser.add_argument(
        "-l",
        "--lambda",
        dest="lambda_",
        type=float,
        default=config.lambda_,
        help="regularization constant for the estimator",
    )
    parser.add_argument(
        "-N",
        "--num_samples",
        type=int,
        default=config.num_samples,
        help="number of samples to use to train the estimator. Ignored if x_samples is provided",
    )
    parser.add_argument(
        "--sigma_f",
        type=float,
        default=1.0,
        help="hyperparameter for the feature map, sigma_f",
    )
    parser.add_argument(
        "--sigma_l",
        default=config.sigma_l,
        type=float,
        action=FloatOrNVectorAction,
        help="hyperparameter for the feature map, sigma_l",
    )
    parser.add_argument(
        "-T",
        "--time_horizon",
        type=int,
        default=config.time_horizon,
        help="time horizon for the scenario",
    )
    parser.add_argument(
        "-f",
        "--num_frequencies",
        type=int,
        default=config.num_frequencies,
        help="number of frequencies per dimension for the Fourier feature map",
    )
    parser.add_argument(
        "--oversample_factor",
        type=float,
        default=config.oversample_factor,
        help="factor by which to oversample the frequency space. If `num_oversample` is set, it takes precedence",
    )
    parser.add_argument(
        "--num_oversample",
        type=int,
        default=config.num_oversample,
        help="number of samples to use for the frequency space. If negative, it is computed based on the oversample factor",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=config.noise_scale,
        help="variance of the additive Gaussian process noise to be added to the system dynamics. If 0, no noise is added. If data is used, this parameter is ignored.",
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
        default=config.problem_log_file,
        help="file to save the optimization problem in LP format. If empty, the problem will not be saved.",
    )
    parser.add_argument(
        "--iis_log_file",
        type=str,
        default=config.iis_log_file,
        help="file to save the irreducible infeasible set (IIS) in ILP format. If empty, the IIS will not be saved.",
    )
    parser.add_argument(
        "--system_dynamics",
        type=str,
        default=config.system_dynamics,
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
        default=config.X_bounds,
        help="state space X bounds as a string. For example, `--X_bounds 'RectSet([-3, -2], [2.5, 1])'`",
    )
    parser.add_argument(
        "--X_init",
        type=type_set,
        default=config.X_init,
        help="initial set X_0 as a string. "
        "For example, `--X_init 'MultiSet(RectSet([1, -0.5], [2, 0.5]), RectSet([-1.8, -0.1], [-1.2, 0.1]))'`",
    )
    parser.add_argument(
        "--X_unsafe",
        type=type_set,
        default=config.X_unsafe,
        help="unsafe set X_U as a string. "
        "For example, `--X_unsafe 'MultiSet(RectSet([0.4, 0.1], [0.6, 0.5]), RectSet([0.4, 0.1], [0.8, 0.3]))'`",
    )
    parser.add_argument(
        "--estimator",
        action=EstimatorAction,
        default=config.estimator,
        choices=["KernelRidgeRegressor"],
        help="estimator type to use. Currently only 'KernelRidgeRegressor' is supported",
    )
    parser.add_argument(
        "--kernel",
        action=KernelAction,
        default=config.kernel,
        choices=["GaussianKernel"],
        help="kernel type to use for the estimator. Currently only 'GaussianKernel' is supported",
    )
    parser.add_argument(
        "--feature_map",
        action=FeatureMapAction,
        default=config.feature_map,
        choices=[
            "LinearTruncatedFourierFeatureMap",
            "ConstantTruncatedFourierFeatureMap",
            "LogTruncatedFourierFeatureMap",
        ],
        help="feature map type to use for the estimator",
    )
    parser.add_argument(
        "--optimiser",
        action=OptimiserAction,
        default=config.optimiser,
        choices=[
            "GurobiOptimiser",
            "AlglibOptimiser",
            "HighsOptimiser",
        ],
        help="feature map type to use for the estimator",
    )

    return parser
