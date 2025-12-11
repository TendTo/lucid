#!/usr/bin/env python3
import numpy as np

from pylucid import *
from pylucid import __version__
from pylucid.pipeline import OptimiserResult, pipeline


def optimiser_cb(res: OptimiserResult):
    """Callback function to handle the results of the optimiser."""
    assert res["success"], "Optimisation failed"
    assert res["obj_val"] <= 0.60, f"Objective value should be <= 0.60, got {res['obj_val']}"
    assert len(res["sol"]) == 97, f"Expected solution length of 97, got {len(res['sol'])}"


def scenario_config() -> "Configuration":
    config = Configuration.from_file("tests/config/barrier2.yaml")
    random.seed(config.seed)
    np.random.seed(config.seed)
    config.populate_samples()
    return config


def test_scenario_config() -> "Configuration":
    """Run the scenario configuration for testing purposes."""
    pipeline(scenario_config(), show=False, optimiser_cb=optimiser_cb)


if __name__ == "__main__":
    test_scenario_config()
