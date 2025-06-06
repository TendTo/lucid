from pylucid import __version__, __doc__, log_error, log_info, set_verbosity, ScenarioConfig
from pylucid.pipeline import pipeline
from typing import TYPE_CHECKING
from argparse import ArgumentParser, Namespace
from pathlib import Path
import importlib
import time

if TYPE_CHECKING:
    from collections.abc import Sequence


def raise_error(message: str, error_type: type = ValueError):
    """Raise an error with the given message."""
    log_error(message)
    return error_type(message)


class CLIArgs(Namespace):
    verbosity: int
    input: Path
    silent: bool


def arg_parser() -> "ArgumentParser":
    def valid_path(path_str: str) -> Path:
        path = Path(path_str)
        if not path.exists():
            raise raise_error(f"Path does not exist: {path_str}")
        if not path.is_file() or not path.suffix == ".py":
            raise raise_error(f"Path must be a Python file: {path_str}")
        return path

    parser = ArgumentParser(prog="pylucid", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("input", help="path to the input file", type=valid_path)
    parser.add_argument(
        "-v", "--verbosity", action="count", default=3, help="increase verbosity (can be used multiple times)"
    )
    parser.add_argument("-s", "--silent", action="store_true", help="suppress all output except errors")

    return parser


def main(argv: "Sequence[str] | None" = None):
    args = arg_parser().parse_args(argv, namespace=CLIArgs())
    # Set verbosity based on the command line argument
    set_verbosity(args.verbosity)
    if args.silent:  # If silent mode is enabled, set verbosity to 0
        set_verbosity(0)
    # Import the input file as a module
    mod = importlib.import_module(".".join(args.input.parts).removesuffix(".py"))
    # Check if the module has a 'scenario_config' function
    if not hasattr(mod, "scenario_config"):
        raise raise_error("The configuration file must contain a function called 'scenario_config'")
    # Validate the configuration returned by the 'scenario_config' function
    config: ScenarioConfig = mod.scenario_config()
    if not isinstance(config, (ScenarioConfig, dict)):
        raise raise_error("The 'scenario_config' function must return an instance of 'ScenarioConfig' or a dict")

    # If all the checks pass, run the benchmark
    log_info(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()
    pipeline(**config)
    end = time.time()
    log_info(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    main()
