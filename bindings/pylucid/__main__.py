import importlib
import inspect
import time

import numpy as np
from pylucid import *
from pylucid import __version__

from .cli import CLIArgs, ScenarioConfig, arg_parser, cli_scenario_config


def main(argv: "Sequence[str] | None" = None):
    args = arg_parser().parse_args(argv, namespace=CLIArgs())
    args.sigma_l = np.array(args.sigma_l) if len(args.sigma_l) > 1 else args.sigma_l[0]
    # Set verbosity based on the command line argument
    set_verbosity(args.verbose)

    if args.input.name == "":
        # If no input file is provided, use the default scenario configuration
        log_info("No input file provided, using default scenario configuration")
        config = cli_scenario_config(args)
    else:
        log_info(f"Loading scenario configuration from file '{args.input}'")
        # Import the input file as a module
        mod = importlib.import_module(".".join(args.input.parts).removesuffix(".py"))
        # Check if the module has a 'scenario_config' function
        if not hasattr(mod, "scenario_config") or not callable(mod.scenario_config):
            raise raise_error("The configuration file must contain a function called 'scenario_config'")
        # Validate the configuration returned by the 'scenario_config' function
        sign = inspect.signature(mod.scenario_config)
        has_params = any(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL) for p in sign.parameters.values()
        )
        if has_params:
            config: ScenarioConfig = mod.scenario_config(args)
        else:
            config: ScenarioConfig = mod.scenario_config()
        if not isinstance(config, (ScenarioConfig, dict)):
            raise raise_error("The 'scenario_config' function must return an instance of 'ScenarioConfig' or a dict")

    # If all the checks pass, run the scenario
    from pylucid.pipeline import pipeline

    log_info(f"Running scenario (LUCID version: {__version__})")
    start = time.time()
    pipeline(**config)
    end = time.time()
    log_info(f"Elapsed time: {end - start}")


if __name__ == "__main__":
    main()
