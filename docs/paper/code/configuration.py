from pylucid import *
import numpy as np

def scenario_config() -> Configuration:
    np.random.seed(42)
    random.seed(42)
    c = Configuration()
    c.seed = 42
    c.X_bounds = RectSet([(-1, 1)])
    c.X_init = RectSet([(-0.5, 0.5)])
    c.X_unsafe = MultiSet(
        RectSet([(-1, -0.9)]), 
        RectSet([(0.9, 1)]),
    )
    c.x_samples = np.array([[1.0], [-1.0]])
    c.xp_samples = np.array([[0.5], [-0.5]])
    c.lambda_ = 1.0e-5
    c.time_horizon = 15
    c.sigma_f = 18.0
    c.sigma_l = 0.034
    c.num_frequencies = 5
    c.num_oversample = 700
    return c
