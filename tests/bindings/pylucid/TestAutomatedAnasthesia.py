import sys
import math
import numpy as np
from pylucid import (
    __version__,
    read_matrix,
    GaussianKernel,
    ConstantTruncatedFourierFeatureMap,
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


def test_automated_anesthesia():
    """
    Automated Anaesthesia Delivery System Benchmark

    The concentration of Propofol in different compartments of the body are modelled using the three-compartment pharmacokinetic system.
    An automated system controls the basal dosage of Propofol to the patient, while an anaesthesiologist can administer a bolus dose.
    The system model includes a stochastic model of the anaesthesiologist behavior based on the concentration and the number of bolus
    doses administered. The hybrid system behavior arises from this anaesthesiologist behavior.

    First presented in Abate, A., Blom, H., Cauchi, N., Hartmanns, A., Lesser, K., Oishi, M., ... & Vinod, A. P. (2018). ARCH-COMP18 category report: Stochastic modelling. In 5th International Workshop on Applied Verification of Continuous and Hybrid Systems, ARCH 2018 (pp. 71-103). EasyChair.

    ## Mathematical Model

    $$
    \\begin{aligned}
        \\bar{x}[k + 1] &= \\begin{bmatrix} 0.8192 & 0.03412 & 0.01265 \\\\ 0.01646 & 0.9822 & 0.0001 \\\\ 0.0009 & 0.00002 & 0.9989 \\end{bmatrix} \\bar{x}[k] + \\begin{bmatrix} 0.01883 \\\\ 0.0002 \\\\ 0.00001 \\end{bmatrix} (v[k] + \\sigma[k]) + w[k] \\\\
                        &= A \\bar{x}[k] + B (v[k] + \\sigma[k]) + w[k]
    \\end{aligned}
    $$

    where $\\bar{x}[k]$ is the continuous state vector, $v[k]$ is the automated delivery system control input, $w[k] \\sim \\mathcal{N}(0, M)$ is the process noise, and $\\sigma[k]$ is the noise from the anaesthesiologist.
    $\\sigma[k]$ is a semi-Markov random variable taking values in $\\{0, 30\\}$ that represents the anaesthesiologist's decision to administer a bolus dose.
    To make the stochastic system Markovian, a binary state vector $q[k]$ is introduced that represents the anaesthesiologist's decision to administer a bolus dose at time $k$.
    Then the stochastic decision to administer a bolus dose is conditioned on $\\bar{x}_1[k]$ and $q[k]$.
    """
    ######## System dynamics ########

    # Deterministic part of the linear dynamics x[t+1] = A * x[t]
    A = np.array(
        [
            [0.8192, 0.03412, 0.01265],
            [0.01646, 0.9822, 0.0001],
            [0.0009, 0.00002, 0.9989],
        ]
    )
    f_det = lambda x: A @ x
    dim = 3  # Dimensionality of the state space

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
    x_samples: "np.typing.ArrayLike" = X_bounds.sample_element(N)
    xp_samples = f(x_samples.T).T
    n_per_dim = factor * samples_per_dim
    sigma_f, sigma_l = median_heuristic(x_samples, x_samples)
    sigma_f /= 2.0
    print(f"Median heuristic: {sigma_f = }, {sigma_l = }")

    k = GaussianKernel(sigma_f, sigma_l)
    tffm = ConstantTruncatedFourierFeatureMap(num_freq_per_dim, sigma_l, sigma_f, X_bounds)
    x_lattice = X_bounds.lattice(samples_per_dim)
    f_lattice = tffm(x_lattice)
    fp_samples = tffm(xp_samples)
    r = GaussianKernelRidgeRegression(k, x_samples, fp_samples, regularization_constant=1e-6)
    if_lattice = r(x_lattice)
    w_mat = np.zeros((n_per_dim**dim, fp_samples.shape[1]))
    phi_mat = np.zeros((n_per_dim**dim, fp_samples.shape[1]))
    for i in range(w_mat.shape[1]):
        w_mat[:, i] = fft_upsample(if_lattice[:, i], samples_per_dim, n_per_dim, dim)
        phi_mat[:, i] = fft_upsample(f_lattice[:, i], samples_per_dim, n_per_dim, dim)

    x0_lattice = X_init.lattice(n_per_dim - 1, True)
    xu_lattice = X_unsafe.lattice(n_per_dim - 1, True)

    f0_lattice = tffm(x0_lattice)
    fu_lattice = tffm(xu_lattice)

    o = GurobiLinearOptimiser(T, gamma, 0, 1, 1, sigma_f)

    def check_cb(
        success: bool, obj_val: float, sol: "np.typing.NDArray[np.float64]", eta: float, c: float, norm: float
    ):
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


if __name__ == "__main__":
    import time

    start = time.time()
    test_automated_anesthesia()
    end = time.time()
    print("elapsed time:", end - start)
