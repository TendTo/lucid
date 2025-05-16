import numpy as np
from pylucid import (
    __version__,
    GaussianKernel,
    ConstantTruncatedFourierFeatureMap,
    TruncatedFourierFeatureMap,
    LogTruncatedFourierFeatureMap,
    LinearTruncatedFourierFeatureMap,
    RectSet,
    MultiSet,
    KernelRidgeRegressor,
    fft_upsample,
    GurobiLinearOptimiser,
    LucidNotSupportedException,
    GUROBI_BUILD,
    set_verbosity,
    LOG_DEBUG,
)
from pylucid.plot import plot_solution
from scipy.spatial.distance import cdist

np.set_printoptions(linewidth=200, suppress=True)

# set_verbosity(LOG_DEBUG)


def rmse(x: "np.typing.NDArray[np.float64]", y: "np.typing.NDArray[np.float64]", ax=0):
    return np.sqrt(((x - y) ** 2).mean(axis=ax))


# from cvc5.pythonic import Real, solve, Solver, sat, Cosine, Sine, ArithRef, BoolVal, And, Or, Implies, ExprRef, Not
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
    regressor: KernelRidgeRegressor,
    r_features: KernelRidgeRegressor,
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
    N = 1000

    # Kernel Basis
    num_freq_per_dim = 8  # Number of frequencies per dimension. Includes the zero frequency.

    ######## Lucid ########
    print(f"Running anesthesia benchmark (LUCID version: {__version__})")

    samples_per_dim = 2 * num_freq_per_dim
    x_samples: "np.typing.ArrayLike" = X_bounds.sample(N)
    xp_samples: "np.typing.ArrayLike" = f(x_samples.T).T
    n_per_dim = samples_per_dim * 2
    print(f"{n_per_dim = }, {samples_per_dim = }")
    sigma_f, sigma_l = median_heuristic(x_samples, x_samples)
    # sigma_f, sigma_l = 1, np.array([3.0]) # Works for the standard kernel (no explicit tffm)
    sigma_f, sigma_l = 1, np.array([15.0])
    print(f"Median heuristic: {sigma_f = }, {sigma_l = }")

    k = GaussianKernel(sigma_f, sigma_l)
    tffm = LogTruncatedFourierFeatureMap(num_freq_per_dim, sigma_l, sigma_f, X_bounds)
    print(tffm.omega.shape, tffm.weights.shape)
    print(tffm.omega, tffm.weights)
    print(tffm.dimension, tffm.num_frequencies)

    x_lattice = X_bounds.lattice(samples_per_dim)
    f_x_lattice = tffm(x_lattice)
    f_xp_samples = tffm(xp_samples)  # Used to train the f_xp regressor

    f_xp_regressor = KernelRidgeRegressor(k, x_samples, f_xp_samples, regularization_constant=1e-6)
    xp_regressor = KernelRidgeRegressor(k, x_samples, xp_samples, regularization_constant=1e-6)

    # values = tffm(f(x_lattice.T).T)
    # for i in range(values.shape[1]):
    #     plt.figure()
    #     plt.plot(xp_lattice, tffm(f(x_lattice.T).T)[:, i], color="green", linestyle="--", label="tffm(xp)")
    #     plt.plot(xp_lattice, f_xp_regressor(x_lattice)[:, i], color="red", linestyle=":", label="f_xp_regressor(x)")
    #     plt.plot(
    #         xp_lattice,
    #         f_xp_regressor(x_lattice, tffm)[:, i],
    #         color="purple",
    #         linestyle="dotted",
    #         label="f_xp_regressor(xp) via approx. regression",
    #     )
    # plt.figure()
    # plt.plot(x_lattice, f_det(x_lattice.T).T, color="green", linestyle="--", label="f(x)")
    # plt.plot(x_lattice, xp_regressor(x_lattice), color="red", linestyle=":", label="xp_regressor(x)")
    # plt.plot(
    #     x_lattice,
    #     xp_regressor(x_lattice, tffm),
    #     color="purple",
    #     linestyle="dotted",
    #     label="xp_regressor(xp) via approx. regression",
    # )
    # plt.legend()
    # plt.show()
    # exit(1)

    f_xp_lattice_via_regressor = f_xp_regressor(x_lattice)

    if False:
        w_mat = np.zeros((n_per_dim**dim, f_xp_samples.shape[1]))
        phi_mat = np.zeros((n_per_dim**dim, f_xp_samples.shape[1]))
        for i in range(w_mat.shape[1]):
            w_mat[:, i] = fft_upsample(
                f_xp_lattice_via_regressor[:, i], to_num_samples=n_per_dim, from_num_samples=samples_per_dim, dimension=dim
            )
            phi_mat[:, i] = fft_upsample(
                f_x_lattice[:, i], to_num_samples=n_per_dim, from_num_samples=samples_per_dim, dimension=dim
            )
    else:
        w_mat = f_xp_regressor(X_bounds.lattice(n_per_dim - 1, True))
        phi_mat = tffm(X_bounds.lattice(n_per_dim - 1, True))


    x0_lattice = X_init.lattice(n_per_dim - 1, True)
    xu_lattice = X_unsafe.lattice(n_per_dim - 1, True)

    f_x0_lattice = tffm(x0_lattice)
    f_xu_lattice = tffm(xu_lattice)

    new_data = X_bounds.sample(500)
    print(f"RMSE on state space (x -> xp) [training data] {rmse(xp_regressor(x_samples), xp_samples)}")
    print(f"RMSE on state space (x -> xp) [new data] {rmse(xp_regressor(new_data), f_det(new_data.T).T)}")
    print(f"RMSE on Fourier features (x -> tffm(xp)) [training data] {rmse(f_xp_regressor(x_samples), f_xp_samples)}")
    print(
        f"RMSE on Fourier features (x -> tffm(xp)) [new data] {rmse(f_xp_regressor(new_data), tffm(f_det(new_data.T).T))}"
    )
    print(
        f"RMSE on Fourier features (x -> tffm(xp)) [training data] {rmse(f_xp_regressor(x_samples, tffm), f_xp_samples)}"
    )
    print(
        f"RMSE on Fourier features (x -> tffm(xp)) [new data] {rmse(f_xp_regressor(new_data, tffm), tffm(f_det(new_data.T).T))}"
    )

    o = GurobiLinearOptimiser(T, gamma, 0, 1, b_kappa=1, sigma_f=sigma_f)

    def check_cb(
        success: bool, obj_val: float, sol: "np.typing.NDArray[np.float64]", eta: float, c: float, norm: float
    ):
        assert success
        print(f"Result: {success = } | {obj_val = } | {eta = } | {c = } | {norm = }\n{sol = }")
        # print("H", f_xp_regressor.coefficients.shape, f_xp_regressor.coefficients)
        # print("K X H", f_xp_regressor(new_data).shape, f_xp_regressor(new_data))
        # print("old K X H", f_xp_regressor(new_data).shape, f_xp_regressor(new_data))
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
            regressor=xp_regressor,
            r_features=f_xp_regressor,
            tffm=tffm,
            sol=sol,
        )
        plot_solution(
            X_bounds=X_bounds,
            X_init=X_init,
            X_unsafe=X_unsafe,
            feature_map=tffm,
            eta=eta,
            gamma=gamma,
            sol=sol,
            f=f_det,
            regressor=f_xp_regressor,
        )
        assert success
        exit(1)

    try:
        assert o.solve(
            f_x0_lattice,
            f_xu_lattice,
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
