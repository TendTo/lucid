import importlib
import inspect
import sys
import time

import numpy as np

from pylucid import *
from pylucid import __version__


def scenario_config(config: Configuration) -> Configuration:
    """
    Default scenario configuration function for CLI usage.
    This function is called when no input file is provided.
    """
    assert_or_raise(
        all((config.X_bounds, config.X_init, config.X_unsafe)),
        "'X_bounds', 'X_init', and 'X_unsafe' must be specified",
    )

    if len(config.x_samples) == 0:
        # If x_samples is not provided, sample it from the bounds
        config.x_samples = config.X_bounds.sample(config.num_samples)
    if len(config.xp_samples) == 0:
        assert_or_raise(
            config.system_dynamics is not None,
            "If no outputs are provided, 'system_dynamics' must be specified",
        )
        # If xp_samples is not provided, compute it using the system dynamics function
        # Noisy system dynamics
        f = lambda x: config.system_dynamics(x) + np.random.normal(scale=config.noise_scale)
        config.xp_samples = f(config.x_samples)

    assert_or_raise(len(config.x_samples) > 0, "No samples to use for the scenario")
    assert_or_raise(len(config.xp_samples) > 0, "No transition samples to use for the scenario")
    assert_or_raise(config.x_samples.ndim == 2, "x_samples must be a 2D array")
    assert_or_raise(config.xp_samples.ndim == 2, "xp_samples must be a 2D array")
    assert_or_raise(
        config.x_samples.shape[0] == config.xp_samples.shape[0],
        "x_samples and xp_samples must have the same number of samples",
    )

    # Return the scenario configuration
    return config


def main(argv: "Sequence[str] | None" = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        # If no arguments are provided, print the help message and exit
        arg_parser().print_help()
        return 0
    config: Configuration = arg_parser().parse_args(argv, namespace=Configuration())

    # If a seed is provided, set the random seed for reproducibility
    random.seed(config.seed)
    if config.seed >= 0:
        np.random.seed(config.seed)
    # Set verbosity based on the command line argument
    log.set_verbosity(config.verbose)

    if config.input.suffix == ".py":
        log.info(f"Loading scenario configuration from file '{config.input}'")
        # Import the input file as a module
        spec = importlib.util.spec_from_file_location(config.input.stem, config.input)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[config.input.stem] = mod
        spec.loader.exec_module(mod)
        # Check if the module has a 'scenario_config' function
        if not hasattr(mod, "scenario_config") or not callable(mod.scenario_config):
            raise raise_error("The configuration file must contain a function called 'scenario_config'")
        # Validate the configuration returned by the 'scenario_config' function
        sign = inspect.signature(mod.scenario_config)
        has_params = any(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL) for p in sign.parameters.values()
        )
        if has_params:
            config: Configuration = mod.scenario_config(config)
        else:
            config: Configuration = mod.scenario_config()
        if not isinstance(config, Configuration):
            raise raise_error("The 'scenario_config' function must return an instance of 'Configuration'")
    else:
        # If no input file is provided, use the default scenario configuration
        log.info("No input file provided, using default scenario configuration")
        config = scenario_config(config)

    # If all the checks pass, run the scenario
    from pylucid.pipeline import pipeline

    log.info(f"Running scenario (LUCID version: {__version__})")
    start = time.time()
    pipeline(config)
    end = time.time()
    log.info(f"Elapsed time: {end - start}")
    return 0


if __name__ == "__main__":
    exit(main())
