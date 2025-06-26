import importlib
import inspect
import sys
import time

import numpy as np

from pylucid import *
from pylucid import __version__


def cli_scenario_config(args: CLIArgs) -> ScenarioConfig:
    """
    Default scenario configuration function for CLI usage.
    This function is called when no input file is provided.
    """
    if not all((args.system_dynamics, args.X_bounds, args.X_init, args.X_unsafe)):
        raise raise_error(
            "If no input file is provided, 'system_dynamics', 'X_bounds', 'X_init', and 'X_unsafe' must be specified"
        )

    # Define the system dynamics function
    f_det = args.system_dynamics
    f = lambda x: f_det(x) + np.random.normal(scale=args.noise_scale)  # Add noise to the dynamics

    # Sample points from the bounds
    x_samples = args.X_bounds.sample(args.num_samples)
    xp_samples = f(x_samples)

    # Create the estimator
    estimator = args.estimator(
        kernel=args.kernel(sigma_f=args.sigma_f, sigma_l=args.sigma_l),
        regularization_constant=args.lambda_,
    )

    # Return the scenario configuration
    return ScenarioConfig(
        x_samples=x_samples,
        xp_samples=xp_samples,
        X_bounds=args.X_bounds,
        X_init=args.X_init,
        X_unsafe=args.X_unsafe,
        T=args.time_horizon,
        gamma=args.gamma,
        num_freq_per_dim=args.num_frequencies,
        f_det=f_det,
        estimator=estimator,
        feature_map=args.feature_map,
        sigma_f=args.sigma_f,
        oversample_factor=args.oversample_factor,
        problem_log_file=args.problem_log_file,
    )


def main(argv: "Sequence[str] | None" = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        # If no arguments are provided, print the help message and exit
        arg_parser().print_help()
        return 0
    args = arg_parser().parse_args(argv, namespace=CLIArgs())
    # If a seed is provided, set the random seed for reproducibility
    random.seed(args.seed)
    if args.seed >= 0:
        np.random.seed(args.seed)
    # Set verbosity based on the command line argument
    log.set_verbosity(args.verbose)

    if args.input.suffix == ".py":
        log.info(f"Loading scenario configuration from file '{args.input}'")
        # Import the input file as a module
        spec = importlib.util.spec_from_file_location(args.input.stem, args.input)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[args.input.stem] = mod
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
            config: ScenarioConfig = mod.scenario_config(args)
        else:
            config: ScenarioConfig = mod.scenario_config()
        if not isinstance(config, (ScenarioConfig, dict)):
            raise raise_error("The 'scenario_config' function must return an instance of 'ScenarioConfig' or a dict")
    else:
        # If no input file is provided, use the default scenario configuration
        log.info("No input file provided, using default scenario configuration")
        config = cli_scenario_config(args)

    # If all the checks pass, run the scenario
    from pylucid.pipeline import pipeline

    log.info(f"Running scenario (LUCID version: {__version__})")
    start = time.time()
    pipeline(**config)
    end = time.time()
    log.info(f"Elapsed time: {end - start}")
    return 0


if __name__ == "__main__":
    exit(main())
