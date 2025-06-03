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
from pylucid.dreal import build_barrier_expression, build_set_constraint
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt

np.set_printoptions(linewidth=200, suppress=True)

# set_verbosity(LOG_DEBUG)


def rmse(x: "np.typing.NDArray[np.float64]", y: "np.typing.NDArray[np.float64]", ax=0):
    return np.sqrt(((x - y) ** 2).mean(axis=ax))


from dreal import And, Implies, Variable as Real, Not, CheckSatisfiability


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
    barrier = build_barrier_expression(xs=xs, X_bounds=X_bounds, tffm=tffm, sigma_f=sigma_f, sol=sol)
    barrier_p = build_barrier_expression(xs=xsp, X_bounds=X_bounds, tffm=tffm, sigma_f=sigma_f, sol=sol)

    tolerance = 1e-8
    constraints = And(
        # Bounds on the state space (X_bounds) for both initial and successive states
        build_set_constraint(xs, X_bounds),
        build_set_constraint(xsp, X_bounds),
        # Specification
        Not(
            And(
                # Non-negativity of the barrier function (-tolerance)
                barrier >= -tolerance,
                # First condition
                Implies(build_set_constraint(xs, X_init), barrier <= eta),
                # Second condition
                Implies(build_set_constraint(xs, X_unsafe), barrier >= gamma),
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


def case_study_template():
    ######## System dynamics ########

    r_coeff = 0.1

    f_det = lambda x: 1 / 2 * x
    # Add process noise
    np.random.seed(50)  # For reproducibility
    f = lambda x: f_det(x) * (np.random.standard_normal())

    dim = 1  # Dimensionality of the state space

    ######## Safety specification ########

    # Time horizon
    T = 5
    # State space X
    X_bounds = RectSet(((-1, 1),), seed=50)

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

    # Kernel hyperparameters
    sigma_f, sigma_l = 1.0, np.array([0.5])

    # Estimator hyperparameters
    regularization_constant = 1e-6

    ######## Lucid ########
    samples_per_dim = 2 * num_freq_per_dim
    x_samples: "np.typing.ArrayLike" = X_bounds.sample(N)
    xp_samples: "np.typing.ArrayLike" = f(x_samples.T).T
    n_per_dim = samples_per_dim * 2

    tffm = ConstantTruncatedFourierFeatureMap(num_freq_per_dim, sigma_l, sigma_f, X_bounds)

    f_xp_samples = tffm(xp_samples)  # Used to train the f_xp regressor

    f_xp_regressor = KernelRidgeRegressor(
        kernel=GaussianKernel(sigma_l=sigma_l, sigma_f=sigma_f),
        regularization_constant=regularization_constant,
        tuner=None,
    )
    f_xp_regressor.fit(x_samples, f_xp_samples)

    print(f"RMSE on f_xp_samples {rmse(f_xp_regressor(x_samples), f_xp_samples)}")
    print(f"Score on f_xp_regressor {f_xp_regressor.score(x_samples, f_xp_samples)}")

    x_lattice = X_bounds.lattice(n_per_dim, True)
    u_f_x_lattice = tffm(x_lattice)
    u_f_xp_lattice_via_regressor = f_xp_regressor(x_lattice)  # What we want to do
    # We are fixing the zero frequency to the constant value we computed in the feature map
    # If we don't, the regressor has a hard time learning it on the extreme left and right points, because it tends to 0
    u_f_xp_lattice_via_regressor[:, 0] = tffm.weights[0]

    x0_lattice = X_init.lattice(n_per_dim, True)
    f_x0_lattice = tffm(x0_lattice)

    xu_lattice = X_unsafe.lattice(n_per_dim, True)
    f_xu_lattice = tffm(xu_lattice)

    o = GurobiLinearOptimiser(T, gamma, 0, 1, b_kappa=1, sigma_f=sigma_f)

    def check_cb(
        success: bool, obj_val: float, sol: "np.typing.NDArray[np.float64]", eta: float, c: float, norm: float
    ):
        print(f"Result: {success = } | {obj_val = } | {eta = } | {c = } | {norm = }\n{sol = }")
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

        assert success

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
    case_study_template()
    end = time.time()
    print("elapsed time:", end - start)
