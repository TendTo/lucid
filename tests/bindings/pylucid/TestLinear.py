#!/usr/bin/env python3
import numpy as np

from pylucid import *
from pylucid import __version__
from pylucid.pipeline import OptimiserResult, pipeline


def optimiser_cb(res: OptimiserResult):
    """Callback function to handle the results of the optimiser."""
    assert res["success"], "Optimisation failed"
    assert res["obj_val"] <= 0.06, "Safety lower bound should be >= 94%"
    assert res["norm"] <= 0.1, "Norm of the solution should be <= 0.1"
    assert len(res["sol"]) == 9


def scenario_config() -> "Configuration":
    config = Configuration.from_file("tests/config/linear.yaml")
    f = lambda x: config.system_dynamics(x) + np.random.normal(scale=config.noise_scale)
    config.x_samples = config.X_bounds.sample(config.num_samples)
    config.xp_samples = f(config.x_samples)
    return config


def test_scenario_config() -> "Configuration":
    """Run the scenario configuration for testing purposes."""
    pipeline(scenario_config(), show=False, optimiser_cb=optimiser_cb)


if __name__ == "__main__":
    test_scenario_config()
