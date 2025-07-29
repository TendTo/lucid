from ScenarioTuner import scenario_tuner, Parameter
import numpy as np
import itertools

if __name__ == "__main__":
    # Define the grid for tuning parameters
    val = np.logspace(-5, 1, num=100, endpoint=True, dtype=float)
    grid = {
        Parameter.REGULARIZATION_CONSTANT: [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8],
        Parameter.SIGMA_F: [1.0],
        Parameter.SIGMA_L: [np.array(vs) for vs in itertools.product(val, val)],
    }

    # Call the scenario tuner with the path to the configuration file and the grid
    scenario_tuner("benchmarks/integration/dc_motor.yaml", grid)
