# example.py
from pylucid import *
import numpy as np

def scenario_config() -> Configuration:
    random.seed(42)
    np.random.seed(42)
    # Configuration
    c = Configuration()
    c.gamma = 15.0
    c.time_horizon = 15
    c.sigma_f = 1.0
    c.sigma_l = [1.75555556]
    c.num_frequencies = 5
    c.oversample_factor = 64.0
    # System parameters
    c.system_dynamics = lambda x: x / 2
    c.X_bounds = RectSet([(-1, 1)])
    c.X_init = RectSet([(-0.5, 0.5)])
    c.X_unsafe = MultiSet(RectSet([(-1, -0.9)]), RectSet([(0.9, 1)]))
    # Sampling
    c.x_samples = c.X_bounds.sample(1000)
    c.xp_samples = c.system_dynamics(c.x_samples) + np.random.normal(scale=0.01)
    # Estimator
    c.estimator = KernelRidgeRegressor(
        kernel=GaussianKernel(sigma_f=c.sigma_f, sigma_l=c.sigma_l),
        regularization_constant=1e-6,
    )
    # Execution options
    c.plot = True
    c.verify = True
    c.optimiser = GurobiOptimiser
    return c