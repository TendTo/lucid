#!/usr/bin/env python3
from pylucid import *
from hp_tuning import hp_tuning
import itertools


if __name__ == "__main__":
    val = np.logspace(-3, 2, num=10, endpoint=True, dtype=float)

    grid = {
        Parameter.REGULARIZATION_CONSTANT: [1.0e-6],
        Parameter.SIGMA_F: [1.0],
        Parameter.SIGMA_L: [np.array(x) for x in itertools.product(val, val, val)],
    }

    hp_tuning("benchmarks/integration/overtaking.yaml", grid)
