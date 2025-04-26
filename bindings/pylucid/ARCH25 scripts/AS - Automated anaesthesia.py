from pylucid import __version__
import numpy as np
import time

"""
Automated anaesthesia (AS) benchmark

Assumption: We assume v[k]=0 and \sigma[k]=0, e.g. the case with no control input to create a verification case study.
"""

"""
TODO: Not too sure about in what order dimensions should be defined in the state space!!!
"""

def anesthesia_benchmark():

    ### System dynamics

    # Deterministic part of the linear dynamics x[t+1] = A * x[t]
    A = np.array([[0.8192, 0.03412, 0.01265],
                    [0.01646, 0.9822, 0.0001],
                    [0.0009, 0.00002, 0.9989]])
    f_det = lamda x: A @ x
    dim = 3 # Dimensionality of the state space
   
    # Add process noise
    mean = np.array([0, 0, 0]) # Mean vector
    sigma = np.diag([5, 5, 5]) # Covariance matrix
    f = lambda x: f_det(x) + np.random.multivariate_normal(mean, cov, x.shape[1]).T

    ### Safety specification

    # State space X := [0, 7] × [−1, 11]^2
    X_bounds = np.array([[0, 7], [-1, 11], [-1, 11]])

    # Initial set X_I := [4, 6] × [8, 10]^2
    X_init = np.array([[4, 6], [8, 10], [8, 10]])

    # Unsafe set X_U := X \ ( [1, 6] × [0, 10]^2 )
    # The set is broken down into 8 hyperrectangular sets
    X_unsafe1 = np.array([[0, 0.9], [-1, -0.1], [-1, -0.1]])
    X_unsafe2 = np.array([[0, 0.9], [-1, -0.1], [10.1, 11]])
    X_unsafe3 = np.array([[0, 0.9], [10.1, 11], [-1, -0.1]])
    X_unsafe4 = np.array([[0, 0.9], [10.1, 11], [10.1, 11]])
    X_unsafe5 = np.array([[6.1, 7], [-1, -0.1], [-1, -0.1]])
    X_unsafe6 = np.array([[6.1, 7], [-1, -0.1], [10.1, 11]])
    X_unsafe7 = np.array([[6.1, 7], [10.1, 11], [-1, -0.1]])
    X_unsafe8 = np.array([[6.1, 7], [10.1, 11], [10.1, 11]])

    X_unsafe = {X_unsafe1, X_unsafe2, X_unsafe3, X_unsafe4, X_unsafe5, X_unsafe6, X_unsafe7, X_unsafe8}

    # Time horizon
    T = 10

    ### TODO: Call LUCID
    print(f"Running anesthesia benchmark (LUCID version: {__version__})")
    return None

# Run
if __name__ == "__main__":
    start = time.time()
    result = anesthesia_benchmark()
    end = time.time()

    print("elapsed time:", end-start)
    if len(result) == 0:
        print("Results dictionary is empty.")
    else:
        print(result)
