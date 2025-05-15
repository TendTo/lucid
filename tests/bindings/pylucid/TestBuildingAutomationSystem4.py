import sys
import math
import numpy as np
from pylucid import (
    __version__,
    GaussianKernel,
    TruncatedFourierFeatureMap,
    RectSet,
    MultiSet,
    GaussianKernelRidgeRegression,
    fft_upsample,
    GurobiLinearOptimiser,
    LucidNotSupportedException,
    GUROBI_BUILD,
)
from scipy.spatial.distance import cdist


def median_heuristic(X, Y):
    """
    the famous kernel median heuristic
    """
    kernel_width = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        distsqr = cdist(X[:, i].reshape(X.shape[0], 1), X[:, i].reshape(X.shape[0], 1), "euclidean") ** 2
        kernel_width[i] = np.sqrt(0.5 * np.median(distsqr))

    """in sklearn, kernel is done by K(x, y) = exp(-gamma ||x-y||^2)"""
    distsqr = cdist(X, X, "euclidean") ** 2
    all_width = np.sqrt(0.5 * np.median(distsqr))
    kernel_gamma = 1.0 / (2 * all_width**2)

    return kernel_gamma, kernel_width


def test_building_automation_system():
    """
    Building Automation System 2D Benchmark

    A building automation system (BAS) with two zones, each heated by one radiator and with a shared air supply.

    First presented in Abate, A., Blom, H., Cauchi, N., Hartmanns, A., Lesser, K., Oishi, M., ... & Vinod, A. P. (2018). ARCH-COMP19 category report: Stochastic modelling. In 5th International Workshop on Applied Verification of Continuous and Hybrid Systems, ARCH 2018 (pp. 71-103). EasyChair.
    Concrete values taken from Abate, A., Blom, H., Cauchi, N., Degiorgio, K., Fraenzle, M., Hahn, E. M., ... & Vinod, A. P. (2019). ARCH-COMP19 category report: Stochastic modelling. In 6th International Workshop on Applied Verification of Continuous and Hybrid Systems, ARCH 2019 (pp. 62-102). EasyChair.

    ## Mathematical Model
    
    $$
    \\begin{aligned}
        x[k + 1] &= A x[k] + B u[k] + Q + B_w w[k]\\\\
        y[k] &= \\begin{bmatrix} 1 & 0 & 0 & 0 \\\\ 0 & 1 & 0 & 0 \\end{bmatrix} x[k]
    \\end{aligned}
    $$
    """
    ######## System dynamics ########

    A = np.array(
        (
            (0.6682, 0.0, 0.02632, 0.0),
            (0.0, 0.6830, 0.0, 0.02096),
            (1.0005, 0.0, -0.000499, 0.0),
            (0.0, 0.8004, 0.0, 0.1996),
        )
    )
    B = np.array((0.1320, 0.1402, 0.0, 0.0))
    Q = np.array((3.3378, 2.9272, 13.0207, 10.4166))
    # Deterministic part of the linear dynamics x[k + 1] = A x[k] + B u[k] + Q
    # u[k] is the control input and it is set to 0
    f_det = lambda x: A @ x + Q  # + B * u

    dim = 4  # Dimensionality of the state space

    # Add process noise
    mean = np.array([0, 0, 0])  # Mean vector
    cov = np.diag([5, 5, 5])  # Covariance matrix
    f = lambda x: f_det(x) + np.random.multivariate_normal(mean, cov, x.shape[1]).T

    ######## Safety specification ########

    # Time horizon
    T = 10
    # State space X := [0, 7] × [−1, 11]^2
    X_bounds = RectSet([[0, 7], [-1, 11], [-1, 11]])
    # np.array([[0, 7], [-1, 11], [-1, 11]])

    # Initial set X_I := [4, 6] × [8, 10]^2
    X_init = RectSet([[4.5, 5.5], [5, 7], [5, 7]])

    # Unsafe set X_U := X \ ( [1, 6] × [0, 10]^2 )
    # The set is broken down into 8 hyperrectangular sets
    X_unsafe = MultiSet(
        RectSet([[0, 0.9], [-1, -0.1], [-1, -0.1]]),
        RectSet([[0, 0.9], [-1, -0.1], [10.1, 11]]),
        RectSet([[0, 0.9], [10.1, 11], [-1, -0.1]]),
        RectSet([[0, 0.9], [10.1, 11], [10.1, 11]]),
        RectSet([[6.1, 7], [-1, -0.1], [-1, -0.1]]),
        RectSet([[6.1, 7], [-1, -0.1], [10.1, 11]]),
        RectSet([[6.1, 7], [10.1, 11], [-1, -0.1]]),
        RectSet([[6.1, 7], [10.1, 11], [10.1, 11]]),
    )

    ######## Parameters ########
    gamma = 1
    eta = 1.1e-6
    c = 1.25e-6
    N = 5000

    # Kernel Basis
    num_supp_per_dim = 8
    num_freq_per_dim = 4

    ######## Lucid ########
    print(f"Running anesthesia benchmark (LUCID version: {__version__})")

    samples_per_dim = 2 * num_freq_per_dim
    factor = math.ceil(num_supp_per_dim / samples_per_dim) + 1
    x_samples: "np.typing.ArrayLike" = X_bounds.sample(N)
    xp_samples = f(x_samples.T).T
    n_per_dim = factor * samples_per_dim
    sigma_f, sigma_l = median_heuristic(x_samples, x_samples)
    sigma_f /= 2.0
    print(f"Median heuristic: {sigma_f = }, {sigma_l = }")

    k = GaussianKernel(sigma_f, sigma_l)
    tffm = TruncatedFourierFeatureMap(num_freq_per_dim, dim, sigma_l, sigma_f, X_bounds)
    x_lattice = X_bounds.lattice(samples_per_dim)
    f_lattice = tffm(x_lattice)
    fp_samples = tffm(xp_samples)
    r = GaussianKernelRidgeRegression(k, x_samples, fp_samples, regularization_constant=1e-6)
    if_lattice = r(x_lattice)
    w_mat = np.zeros((n_per_dim**dim, fp_samples.shape[1]))
    phi_mat = np.zeros((n_per_dim**dim, fp_samples.shape[1]))
    for i in range(w_mat.shape[1]):
        w_mat[:, i] = fft_upsample(if_lattice[:, i], n_per_dim, samples_per_dim, dim)
        phi_mat[:, i] = fft_upsample(f_lattice[:, i], n_per_dim, samples_per_dim, dim)

    x0_lattice = X_init.lattice(n_per_dim - 1, True)
    xu_lattice = X_unsafe.lattice(n_per_dim - 1, True)

    f0_lattice = tffm(x0_lattice)
    fu_lattice = tffm(xu_lattice)

    o = GurobiLinearOptimiser(T, gamma, 0, 1, 1, sigma_f)

    def check_cb(success: bool, obj_val: float, eta: float, c: float, norm: float):
        print(f"Result: {success = } | {obj_val = } | {eta = } | {c = } | {norm = }")
        assert success

    try:
        assert o.solve(
            f0_lattice,
            fu_lattice,
            phi_mat,
            w_mat,
            tffm.dimension,
            num_freq_per_dim - 1,
            n_per_dim,
            dim,
            check_cb,
        )
        assert GUROBI_BUILD
    except LucidNotSupportedException:
        assert not GUROBI_BUILD  # Did not compile against Gurobi. Ignore this test.

    sys.exit(1)


if __name__ == "__main__":
    import time

    start = time.time()
    test_building_automation_system()
    end = time.time()
    print("elapsed time:", end - start)
