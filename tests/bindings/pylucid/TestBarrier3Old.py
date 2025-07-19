import math

import numpy as np
import pytest

from pylucid import *


def set_contraint(xs: "list[ArithRef]", X_set: "Set") -> "ExprRef":
    from cvc5.pythonic import (
        And,
        ArithRef,
        BoolVal,
        Cosine,
        ExprRef,
        Implies,
        Not,
        Or,
        Real,
        Sine,
        Solver,
        sat,
        solve,
    )

    if isinstance(X_set, RectSet):
        return And(*(b for i, x in enumerate(xs) for b in (x >= X_set.lower_bound[i], x <= X_set.upper_bound[i])))
    if isinstance(X_set, MultiSet):
        expr = None
        for rect in X_set:
            expr = Or(expr, set_contraint(xs, rect)) if expr is not None else set_contraint(xs, rect)
        return expr
    raise ValueError("X_set must be a RectSet or MultiSet.")


def verify_barrier_certificate(
    X_bounds: RectSet,
    X_init: RectSet,
    X_unsafe: MultiSet,
    eta: float,
    gamma: float,
    tffm: TruncatedFourierFeatureMap,
    sol: "np.typing.NDArray[np.float64]",
):
    if __name__ != "__main__":  # only verify if run as script
        return
    from cvc5.pythonic import (
        And,
        ArithRef,
        BoolVal,
        Cosine,
        ExprRef,
        Implies,
        Not,
        Or,
        Real,
        Sine,
        Solver,
        sat,
        solve,
    )

    # Create symbolic variables for the input dimensions
    xs = [Real(f"x{i}") for i in range(X_bounds.dimension)]

    # Encode the truncated Fourier feature map as a symbolic expression in terms of xs
    sym_tffm = [tffm.weights[0]]
    for row in tffm.omega[1:]:
        sym_tffm.append(Cosine(sum(o * x for o, x in zip(row, xs, strict=True))))
        sym_tffm.append(Sine(sum(o * x for o, x in zip(row, xs, strict=True))))
    for i, w in enumerate(tffm.weights):
        sym_tffm[i] *= w

    barrier: ArithRef = sum(sym_t * s for sym_t, s in zip(sym_tffm, sol, strict=True))

    s = Solver()
    # Bounds on the input variables (X_bounds)
    s.add(set_contraint(xs, X_bounds))
    constraints = [
        True,
        # Non-negativity of the barrier function
        barrier >= 0,
        # Implies(set_contraint(xs, X_init), barrier <= eta),
        # Implies(set_contraint(xs, X_unsafe), barrier >= gamma),
    ]
    s.add(Not(And(*constraints)))
    # s.add(Implies(set_contraint(xs, X_init), barrier <= eta))
    # s.add(Implies(set_contraint(xs, X_unsafe), barrier >= gamma))

    # print(s.sexpr(), file=sys.stderr)
    assert sat == s.check()
    m = s.model()
    print("Model:", m)


@pytest.mark.skip(reason="Not implemented")
def test_barrier_3():
    ######## PARAMS ########
    num_supp_per_dim = 12
    dimension = 2
    num_freq_per_dim = 6
    sigma_f = 19.456
    sigma_l = [30, 23.568]
    b_norm = 25
    kappa_b = 1.0
    gamma = 18.312
    T = 10
    lmda = 1e-5
    N = 1000
    epsilon = 1e-3
    autonomous = True

    limit_set = RectSet((-3, -2), (2.5, 1))
    initial_set = MultiSet(
        RectSet((1, -0.5), (2, 0.5)),
        RectSet((-1.8, -0.1), (-1.2, 0.1)),
        RectSet((-1.4, -0.5), (-1.2, 0.1)),
    )
    unsafe_set = MultiSet(RectSet((0.4, 0.1), (0.6, 0.5)), RectSet((0.4, 0.1), (0.8, 0.3)))

    samples_per_dim = 2 * num_freq_per_dim
    factor = math.ceil(num_supp_per_dim / samples_per_dim) + 1
    n_per_dim = factor * samples_per_dim

    x_samples = read_matrix("tests/bindings/pylucid/x_samples.matrix")
    xp_samples = read_matrix("tests/bindings/pylucid/xp_samples.matrix")

    ######## CODE ########
    k = GaussianKernel(sigma_l, sigma_f)
    assert k is not None
    tffm = TruncatedFourierFeatureMap(num_freq_per_dim, dimension, sigma_l, sigma_f, limit_set)
    assert tffm is not None

    x_lattice = limit_set.lattice(samples_per_dim)
    assert x_lattice is not None

    f_lattice = tffm(x_lattice)
    assert f_lattice.shape == (
        samples_per_dim**dimension,
        num_freq_per_dim**dimension * 2 - 1,
    )
    fp_samples = tffm(xp_samples)
    assert fp_samples.shape == (
        xp_samples.shape[0],
        num_freq_per_dim**dimension * 2 - 1,
    )

    r = KernelRidgeRegressor(k, x_samples, fp_samples, lmda)
    assert r is not None

    if_lattice = r(x_lattice)
    assert if_lattice.shape == (144, 71)

    w_mat = np.zeros((n_per_dim**dimension, fp_samples.shape[1]))
    phi_mat = np.zeros((n_per_dim**dimension, fp_samples.shape[1]))
    assert w_mat.shape == (576, 71)
    assert phi_mat.shape == (576, 71)
    for i in range(w_mat.shape[1]):
        w_mat[:, i] = fft_upsample(
            if_lattice[:, i], to_num_samples=n_per_dim, from_num_samples=samples_per_dim, dimension=dimension
        )
        phi_mat[:, i] = fft_upsample(
            f_lattice[:, i], to_num_samples=n_per_dim, from_num_samples=samples_per_dim, dimension=dimension
        )

    x0_lattice = initial_set.lattice(n_per_dim - 1, True)
    assert x0_lattice.shape == (1587, 2)
    xu_lattice = unsafe_set.lattice(n_per_dim - 1, True)
    assert xu_lattice.shape == (1058, 2)

    f0_lattice = tffm(x0_lattice)
    assert f0_lattice.shape == (1587, num_freq_per_dim**dimension * 2 - 1)
    fu_lattice = tffm(xu_lattice)
    assert fu_lattice.shape == (1058, num_freq_per_dim**dimension * 2 - 1)

    o = GurobiOptimiser(T, gamma, epsilon, b_norm, kappa_b, sigma_f)

    def check_cb(
        success: bool, obj_val: float, sol: "np.typing.NDArray[np.float64]", eta: float, c: float, norm: float
    ):
        tolerance = 1e-3
        assert success
        assert math.isclose(obj_val, 0.8375267440200334, rel_tol=tolerance)
        assert math.isclose(eta, 15.336789736494852, rel_tol=tolerance)
        assert math.isclose(c, 0, rel_tol=tolerance)
        assert math.isclose(norm, 10.39392985811301, rel_tol=tolerance)
        print(f"Result: {success = } | {obj_val = } | {eta = } | {c = } | {norm = }")
        verify_barrier_certificate(
            X_bounds=limit_set, X_init=initial_set, X_unsafe=unsafe_set, eta=1, gamma=gamma, tffm=tffm, sol=sol
        )
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
            dimension,
            check_cb,
        )
        assert GUROBI_BUILD
    except exception.LucidNotSupportedException:
        assert not GUROBI_BUILD  # Did not compile against Gurobi. Ignore this test.


if __name__ == "__main__":
    import time

    start = time.time()
    test_barrier_3()
    end = time.time()
    print("elapsed time:", end - start)
