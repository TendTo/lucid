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


def print_solution(
    X_bounds: RectSet,
    X_init: RectSet,
    X_unsafe: MultiSet,
    tffm: TruncatedFourierFeatureMap,
    eta: float,
    gamma: float,
    sol: "np.typing.NDArray[np.float64]",
):
    if __name__ != "__main__":  # only plot if run as script
        return

    import matplotlib.pyplot as plt

    plt.xlim(X_bounds.lower_bound, X_bounds.upper_bound)
    # Draw the unsafe set
    for i in range(len(X_unsafe)):
        unsafe_set = X_unsafe[i]
        plt.plot(
            [unsafe_set.lower_bound, unsafe_set.upper_bound],
            [0, 0],
            color="red",
            label="unsafe set" if i == 0 else "",
        )

    plt.plot(
        [X_init.lower_bound, X_init.upper_bound],
        [0, 0],
        color="blue",
        label="initial set",
    )
    x_lattice = X_bounds.lattice(100)
    f_lattice = tffm(x_lattice)
    values = f_lattice @ sol.T
    plt.plot(x_lattice, values, color="green")
    plt.plot((X_bounds.lower_bound, X_bounds.upper_bound), (eta, eta), color="green", linestyle="dotted", label="eta")
    plt.plot(
        (X_bounds.lower_bound, X_bounds.upper_bound), (gamma, gamma), color="red", linestyle="dotted", label="gamma"
    )
    plt.title("Barrier certificate")
    plt.xlabel("State space")
    plt.legend()
    plt.show()


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
        x[k + 1] &= (1 − \\beta − \\theta \\nu)x[k] + \\theta T_h \\nu + \\beta T_e + R \\zeta
        \\nu &= -0.0120155x + 0.8
    \\end{aligned}
    $$
    """
    ######## System dynamics ########

    th = 45
    te = -15
    r = 0.1
    beta = 0.06
    theta = 0.145

    # Deterministic part of the linear dynamics x[k + 1] = (1 − β − θν)x[k] + θThν + βTe + Rς
    # where ν is -0.0120155x + 0.8
    f_det = lambda x: (1 - beta - theta * -0.0120155 * x + 0.8) * x + theta * th * -0.0120155 * x + 0.8 + beta * te

    dim = 1  # Dimensionality of the state space

    # Add process noise
    f = lambda x: f_det(x) + r * np.random.exponential(1)

    ######## Safety specification ########

    # Time horizon
    T = 5
    # State space X := [1, 50]
    X_bounds = RectSet(((1, 50),))

    # Initial set X_I := [19.5, 20]
    X_init = RectSet(((19.5, 20),))

    # Unsafe set X_U := [1, 17] U [23, 50]
    X_unsafe = MultiSet(
        RectSet(((1, 17),)),
        RectSet(((23, 50),)),
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

    def check_cb(
        success: bool, obj_val: float, sol: "np.typing.NDArray[np.float64]", eta: float, c: float, norm: float
    ):
        print(f"Result: {success = } | {obj_val = } | {eta = } | {c = } | {norm = }")
        print_solution(X_bounds, X_init, X_unsafe, tffm, eta, gamma, sol)
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
    test_building_automation_system()
    end = time.time()
    print("elapsed time:", end - start)
