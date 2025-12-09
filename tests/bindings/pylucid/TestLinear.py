#!/usr/bin/env python3
import numpy as np

from pylucid import *
from pylucid import __version__
from pylucid.pipeline import OptimiserResult, pipeline


def optimiser_cb(res: OptimiserResult):
    """Callback function to handle the results of the optimiser."""
    assert res["success"], "Optimisation failed"
    assert res["obj_val"] <= 0.10, f"Objective value should be <= 0.10, got {res['obj_val']}"
    assert len(res["sol"]) == 11, f"Expected solution length of 11, got {len(res['sol'])}"


def scenario_config() -> "Configuration":
    config = Configuration.from_file("tests/config/linear.yaml")
    random.seed(config.seed)
    np.random.seed(config.seed)
    config.populate_samples()
    return config


def test_scenario_config() -> "Configuration":
    """Run the scenario configuration for testing purposes."""
    pipeline(scenario_config(), show=False, optimiser_cb=optimiser_cb)


if __name__ == "__main__":
    test_scenario_config()
