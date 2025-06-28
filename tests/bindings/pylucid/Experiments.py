import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from pylucid import *
from pylucid.plot import plot_solution

np.set_printoptions(linewidth=200, suppress=True)

# log.set_verbosity(log.LOG_DEBUG)


def rmse(x: "np.typing.NDArray[np.float64]", y: "np.typing.NDArray[np.float64]", ax=0):
    return np.sqrt(((x - y) ** 2).mean(axis=ax))


# from cvc5.pythonic import Real, solve, Solver, sat, Cosine, Sine, ArithRef, BoolVal, And, Or, Implies, ExprRef, Not
from dreal import And, CheckSatisfiability, Implies, Not, Or
from dreal import Variable as Real
from dreal import cos as Cosine
from dreal import sin as Sine


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
    tffm: TruncatedFourierFeatureMap,
    sol: "np.typing.NDArray[np.float64]",
):
    if __name__ != "__main__":  # only verify if run as script
        return

    # Create symbolic variables for the input dimensions
    xs = [Real(f"x{i}") for i in range(X_bounds.dimension)]
    xsp = [f_det(x) for x in xs]
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
        pointp = f_det(point)
        print(f"X: {point}, barrier value: {tffm(point) @ sol.T}")
        print(f"Xp: {pointp}, barrier value: {tffm(pointp) @ sol.T}")
        print(f"Xpemb: ?, barrier value: {regressor(point) @ sol.T}")
        print((tffm(point), regressor(point)))


def experiments():
    ######## System dynamics ########

    r_coeff = 0.1

    f_det = lambda x: 1 / 2 * x
    # Add process noise
    np.random.seed(50)  # For reproducibility
    f = lambda x: f_det(x) * (np.random.standard_normal())
    # f = lambda x: f_det(x) + r_coeff * (np.random.standard_normal())
    # f = lambda x: f_det(x)

    dim = 1  # Dimensionality of the state space

    ######## Safety specification ########

    # Time horizon
    T = 5
    # State space X
    X_bounds = RectSet(((-1, 1),))

    # Initial set X_I
    X_init = RectSet(((-0.5, 0.5),))

    # Unsafe set X_U
    X_unsafe = MultiSet(
        RectSet(((-1, -0.9),)),
        RectSet(((0.9, 1),)),
    )

    ######## Parameters ########
    gamma = 1
    N = 1000

    # Kernel Basis
    num_freq_per_dim = 4  # Number of frequencies per dimension. Includes the zero frequency.

    # sigma_f, sigma_l = 1, np.array([3.0]) # Works for the standard kernel (no explicit tffm)
    sigma_f, sigma_l = 1, np.array([0.5])
    print(f"Median heuristic: {sigma_f = }, {sigma_l = }")

    ######## Lucid ########
    samples_per_dim = 2 * num_freq_per_dim
    x_samples: "np.typing.ArrayLike" = X_bounds.sample(N)
    xp_samples: "np.typing.ArrayLike" = f(x_samples.T).T
    n_per_dim = samples_per_dim * 2

    k = GaussianKernel(sigma_f=sigma_f, sigma_l=sigma_l)
    tffm = ConstantTruncatedFourierFeatureMap(num_freq_per_dim, sigma_l, sigma_f, X_bounds)

    x_lattice = X_bounds.lattice(samples_per_dim)
    f_x_lattice = tffm(x_lattice)
    f_xp_samples = tffm(xp_samples)  # Used to train the f_xp regressor
    f_xp_regressor = KernelRidgeRegressor(k, regularization_constant=1e-6)
    f_xp_regressor.fit(x_samples, f_xp_samples)

    print(f"RMSE on f_xp_samples {rmse(f_xp_regressor(x_samples), f_xp_samples)}")
    print(f"Score on f_xp_regressor {f_xp_regressor.score(x_samples, f_xp_samples)}")
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
    #         label="f_xp_regressor(xp) via regression",
    #     )
    # plt.figure()
    # plt.plot(x_lattice, f_det(x_lattice.T).T, color="green", linestyle="--", label="f(x)")
    # plt.plot(x_lattice, xp_regressor(x_lattice), color="red", linestyle=":", label="xp_regressor(x)")
    # plt.plot(
    #     x_lattice,
    #     xp_regressor(x_lattice, tffm),
    #     color="purple",
    #     linestyle="dotted",
    #     label="xp_regressor(xp) via regression",
    # )
    # plt.legend()
    # plt.show()
    # exit(1)

    if False:
        f_xp_lattice_via_regressor = f_xp_regressor(x_lattice)
        print(
            f"RMSE on f_xp_lattice_via_regressor (x -> tffm(xp)) {rmse(f_xp_lattice_via_regressor, tffm(f_det(x_lattice.T).T))}"
        )
        # We are fixing the zero frequency to the constant value we computed in the feature map
        # If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
        u_f_xp_lattice_via_regressor = np.full((n_per_dim**dim, f_xp_samples.shape[1]), tffm.weights[0])
        u_f_x_lattice = np.full((n_per_dim**dim, f_xp_samples.shape[1]), tffm.weights[0])
        for i in range(1, u_f_xp_lattice_via_regressor.shape[1]):
            u_f_xp_lattice_via_regressor[:, i] = fft_upsample(
                f_xp_lattice_via_regressor[:, i],
                to_num_samples=n_per_dim,
                from_num_samples=samples_per_dim,
                dimension=dim,
            )
            u_f_x_lattice[:, i] = fft_upsample(
                f_x_lattice[:, i], to_num_samples=n_per_dim, from_num_samples=samples_per_dim, dimension=dim
            )
        print(
            f"RMSE on u_f_xp_lattice_via_regressor {rmse(u_f_xp_lattice_via_regressor, tffm(f_det(X_bounds.lattice(n_per_dim).T).T))}"
        )
        print(f"RMSE on u_f_x_lattice {rmse(u_f_x_lattice, tffm(X_bounds.lattice(n_per_dim).T))}")

    else:
        _x_lattice = X_bounds.lattice(n_per_dim, True)

        u_f_xp_lattice_via_regressor = f_xp_regressor(_x_lattice)  # What we want to do
        # We are fixing the zero frequency to the constant value we computed in the feature map
        # If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
        u_f_xp_lattice_via_regressor[:, 0] = tffm.weights[0]

        u_f_x_lattice = tffm(_x_lattice)

    x0_lattice = X_init.lattice(n_per_dim, True)
    xu_lattice = X_unsafe.lattice(n_per_dim, True)

    f_x0_lattice = tffm(x0_lattice)
    f_xu_lattice = tffm(xu_lattice)

    new_data = np.linspace(-1, 1, 1000)
    for i in range(num_freq_per_dim * 2 - 1):
        plt.subplot(num_freq_per_dim * 2 - 1, 1, i + 1)
        plt.scatter(new_data, tffm(f_det(new_data.T).T)[:, i], color="lightblue", linestyle=":", label="tffm(xp)")
        plt.scatter(new_data, tffm(new_data)[:, i], color="lightgreen", linestyle=":", label="tffm(x)")
        plt.scatter(new_data, f_xp_regressor(new_data)[:, i], color="red", marker="+", label="f_xp_regressor(x)")
        plt.title(f"Fourier feature {i}")
    plt.legend()
    plt.show()

    o = GurobiOptimiser(T, gamma, 0, 1, b_kappa=1, sigma_f=sigma_f)

    def check_cb(
        success: bool, obj_val: float, sol: "np.typing.NDArray[np.float64]", eta: float, c: float, norm: float
    ):
        assert success
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
            regressor=f_xp_regressor,
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
            estimator=f_xp_regressor,
        )
        assert success
        exit(1)

    try:
        assert o.solve(
            f_x0_lattice,
            f_xu_lattice,
            u_f_x_lattice,
            u_f_xp_lattice_via_regressor,
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

    print(f"Running benchmark (LUCID version: {__version__})")
    start = time.time()
    experiments()
    end = time.time()
    print("elapsed time:", end - start)
