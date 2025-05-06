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


def rmse(x: "np.typing.NDArray[np.float64]", y: "np.typing.NDArray[np.float64]", ax=0):
    return np.sqrt(((x - y) ** 2).mean(axis=ax))


# from cvc5.pythonic import Real, solve, Solver, sat, Cosine, Sine, ArithRef, BoolVal, And, Or, Implies, ExprRef, Not
import sys
from dreal import And, Or, Implies, Variable as Real, sin as Sine, cos as Cosine, Not, CheckSatisfiability


def set_constraint(xs: "list[ArithRef]", X_set: "RectSet | MultiSet") -> "ExprRef":
    if isinstance(X_set, RectSet):
        return And(*(b for i, x in enumerate(xs) for b in (x >= X_set.lower_bound[i], x <= X_set.upper_bound[i])))
    if isinstance(X_set, MultiSet):
        expr = None
        for rect in X_set:
            expr = Or(expr, set_constraint(xs, rect)) if expr is not None else set_constraint(xs, rect)
        return expr
    raise ValueError("X_set must be a RectSet or MultiSet.")


def barrier_expression(
    xs: "list[ArithRef]",
    X_bounds: "RectSet",
    tffm: "TruncatedFourierFeatureMap",
    sigma_f: float,
    sol: "np.typing.NDArray[np.float64]",
):
    # Encode the truncated Fourier feature map as a symbolic expression in terms of xs
    sym_tffm = [1.0]
    for row in tffm.omega[1:]:
        sym_tffm.append(
            Cosine(
                sum(
                    o / (ub - lb) * (x - lb)
                    for o, x, lb, ub in zip(row, xs, X_bounds.lower_bound, X_bounds.upper_bound, strict=True)
                )
            )
        )
        sym_tffm.append(
            Sine(
                sum(
                    o / (ub - lb) * (x - lb)
                    for o, x, lb, ub in zip(row, xs, X_bounds.lower_bound, X_bounds.upper_bound, strict=True)
                )
            )
        )
    for i, (w, s) in enumerate(zip(tffm.weights, sol, strict=True)):
        sym_tffm[i] *= w * sigma_f * s
    return sum(sym_tffm)


def transition_variables(xs: "list[ArithRef]"):
    # Encode the truncated Fourier feature map as a symbolic expression in terms of xs
    pass


def verify_barrier_certificate(
    X_bounds: RectSet,
    X_init: RectSet,
    X_unsafe: MultiSet,
    sigma_f: float,
    eta: float,
    gamma: float,
    f_det: "callable",
    r: float,
    c: float,
    regressor: GaussianKernelRidgeRegression,
    r_features: GaussianKernelRidgeRegression,
    tffm: TruncatedFourierFeatureMap,
    sol: "np.typing.NDArray[np.float64]",
):
    if __name__ != "__main__":  # only verify if run as script
        return

    # Create symbolic variables for the input dimensions
    xs = [Real(f"x{i}") for i in range(X_bounds.dimension)]
    xsp = [f_det(x) + r * (1 / 1) for x in xs]
    barrier = barrier_expression(xs=xs, X_bounds=X_bounds, tffm=tffm, sigma_f=sigma_f, sol=sol)
    barrier_p = barrier_expression(xs=xsp, X_bounds=X_bounds, tffm=tffm, sigma_f=sigma_f, sol=sol)

    tolerance = 1e-8
    constraints = And(
        # Bounds on the state space (X_bounds) for both initial and successive states
        set_constraint(xs, X_bounds),
        set_constraint(xsp, X_bounds),
        # Specification
        Not(
            And(
                # Non-negativity of the barrier function (-tolerance)
                barrier >= -tolerance,
                # First condition
                Implies(set_constraint(xs, X_init), barrier <= eta),
                # Second condition
                Implies(set_constraint(xs, X_unsafe), barrier >= gamma),
                # Third condition
                barrier_p - barrier <= c,
            ),
        ),
    )
    # print(constraints, file=sys.stderr)
    res = CheckSatisfiability(constraints, 1e-8)
    if res is None:
        print("FEST! The barrier is verified")
    else:
        print("Found counter example")
        print("Model", res)
        point = np.array([res[xs[0]].lb()])
        pointp = f_det(point) + r * np.random.exponential(1)
        pointr = regressor(point)
        print(f"X: {point}, barrier value: {tffm(point) @ sol.T}")
        print(f"Xp: {pointp}, barrier value: {tffm(pointp) @ sol.T}")
        print(f"Xpr: {pointr}, barrier value: {tffm(pointr) @ sol.T}")
        print(f"Xpemb: ?, barrier value: {r_features(point) @ sol.T}")
        print((tffm(point), r_features(point)))


def print_solution(
    X_bounds: RectSet,
    X_init: RectSet,
    X_unsafe: MultiSet,
    tffm: TruncatedFourierFeatureMap,
    r: GaussianKernelRidgeRegression,
    eta: float,
    f: "callable",
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
    xp_lattice = f(x_lattice.T).T
    f_lattice = tffm(x_lattice)
    plt.plot(x_lattice, f_lattice @ sol.T, color="green", label="tffm(x) barrier")
    plt.plot(xp_lattice, r(x_lattice) @ sol.T, color="purple", label="r(x) barrier")
    plt.plot(xp_lattice, tffm(f(x_lattice.T).T) @ sol.T, color="black", label="tffm(f(x)) barrier")
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
    print(f"{kernel_gamma = }, {kernel_width = }")
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
    r_coeff = 0.1
    beta = 0.06
    theta = 0.145

    # Deterministic part of the linear dynamics x[k + 1] = (1 − β − θν)x[k] + θThν + βTe + Rς
    # where ν is -0.0120155x + 0.8
    f_det = lambda x: (1 - beta - theta * -0.0120155 * x + 0.8) * x + theta * th * -0.0120155 * x + 0.8 + beta * te

    dim = 1  # Dimensionality of the state space

    # Add process noise
    f = lambda x: f_det(x) + r_coeff * np.random.exponential(1)

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
    N = 10

    # Kernel Basis
    num_supp_per_dim = 16
    num_freq_per_dim = 8

    ######## Lucid ########
    print(f"Running anesthesia benchmark (LUCID version: {__version__})")

    samples_per_dim = 2 * num_freq_per_dim
    factor = math.ceil(num_supp_per_dim / samples_per_dim) + 1
    x_samples: "np.typing.ArrayLike" = X_bounds.sample_element(N)
    xp_samples: "np.typing.ArrayLike" = f(x_samples.T).T
    n_per_dim = factor * samples_per_dim
    sigma_f, sigma_l = median_heuristic(x_samples, x_samples)
    sigma_f, sigma_l = 1, np.array([10.0])
    print(f"Median heuristic: {sigma_f = }, {sigma_l = }")

    k = GaussianKernel(sigma_f, sigma_l)
    tffm = TruncatedFourierFeatureMap(num_freq_per_dim, dim, sigma_l, sigma_f, X_bounds)
    # tmp = np.square(tffm.weights)
    # tmp[1:] /= 2
    # # print("Weights: ", tmp, sum(tmp))
    # exit(1)
    x_lattice = X_bounds.lattice(samples_per_dim)
    f_lattice = tffm(x_lattice)
    fp_samples = tffm(xp_samples)
    r = GaussianKernelRidgeRegression(k, x_samples, fp_samples, regularization_constant=1e-6)
    regressor_xp = GaussianKernelRidgeRegression(k, x_samples, xp_samples, regularization_constant=1e-6)
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

    new_data = X_bounds.sample_element(500)
    print(f"RMSE on state space (x -> xp) [training data] {rmse(regressor_xp(x_samples), xp_samples)}")
    print(f"RMSE on state space (x -> xp) [new data] {rmse(regressor_xp(new_data), f(new_data.T).T)}")
    print(f"RMSE on Fourier features (x -> tffm(xp)) [training data] {rmse(r(x_samples), fp_samples)}")
    print(f"RMSE on Fourier features (x -> tffm(xp)) [new data] {rmse(r(new_data), tffm(f(new_data.T).T))}")

    o = GurobiLinearOptimiser(T, gamma, 0, 1, 1, sigma_f)

    def check_cb(
        success: bool, obj_val: float, sol: "np.typing.NDArray[np.float64]", eta: float, c: float, norm: float
    ):
        print(f"Result: {success = } | {obj_val = } | {eta = } | {c = } | {norm = }\n{sol = }")
        verify_barrier_certificate(
            X_bounds=X_bounds,
            X_init=X_init,
            X_unsafe=X_unsafe,
            sigma_f=sigma_f,
            eta=eta,
            c=c,
            f_det=f_det,
            r=r_coeff,
            gamma=gamma,
            regressor=regressor_xp,
            r_features=r,
            tffm=tffm,
            sol=sol,
        )
        print_solution(
            X_bounds=X_bounds, X_init=X_init, X_unsafe=X_unsafe, tffm=tffm, eta=eta, gamma=gamma, sol=sol, f=f, r=r
        )
        assert success
        exit(1)

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
