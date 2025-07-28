import math
from typing import TYPE_CHECKING

import numpy as np

from ._pylucid import (
    Estimator,
    MultiSet,
    RectSet,
    SphereSet,
    TruncatedFourierFeatureMap,
    log,
    PolytopeSet,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ._pylucid import NMatrix, NVector, Set

try:
    from dreal import And, CheckSatisfiability, Implies, Not, Or
    from dreal import Variable as Real
    from dreal import cos as Cosine
    from dreal import sin as Sine
except ImportError as e:
    log.warn("Could not import dreal. Make sure it is installed with 'pip install dreal'")
    raise e

Real.cos = lambda self: Cosine(self)
Real.sin = lambda self: Sine(self)

math_original_cos = math.cos
math_original_sin = math.sin
math.cos = lambda x: Cosine(x) if isinstance(x, Real) else math_original_cos(x)
math.sin = lambda x: Sine(x) if isinstance(x, Real) else math_original_sin(x)


def build_set_constraint(xs: "list", X_set: "Set"):
    """
    Generate a set constraint for the given variables to ensure they lie within the bounds of the given set.

    Example:


    Args:
        xs: list of variables (e.g., [x1, x2, ...])
        X_set: a RectSet or MultiSet representing the bounds of the set
        (e.g., RectSet([0, 0], [1, 1]) or MultiSet(RectSet([0, 0], [1, 1]), RectSet([2, 2], [3, 3])))

    Raises:
        ValueError: if X_set is not a supported set type

    Returns:
        Symbolic formula representing the set constraint.
        For example, if X_set is RectSet([0, 0], [1, 1]), it returns (x1 >= 0) && (x1 <= 1) && (x2 >= 0) && (x2 <= 1).
    """
    if isinstance(X_set, RectSet):
        return And(*(b for i, x in enumerate(xs) for b in (x >= X_set.lower_bound[i], x <= X_set.upper_bound[i])))
    if isinstance(X_set, SphereSet):
        return And((sum((x - c) ** 2 for x, c in zip(xs, X_set.center)) <= X_set.radius**2))
    if isinstance(X_set, PolytopeSet):
        A, b = X_set.A, X_set.b
        assert len(A.shape) == 2
        # TODO: double check it is correct
        return And(*(sum(x * A[row][col] for col, x in enumerate(xs)) <= b[row] for row in range(A.shape[0])))
    if isinstance(X_set, MultiSet):
        expr = None
        for rect in X_set:
            expr = Or(expr, build_set_constraint(xs, rect)) if expr is not None else build_set_constraint(xs, rect)
        return expr
    raise ValueError("X_set must be a RectSet or MultiSet.")


def build_barrier_expression(
    xs: "list",
    X_bounds: "RectSet",
    tffm: "TruncatedFourierFeatureMap",
    sigma_f: float,
    sol: "NVector",
):
    """Build a barrier expression using the truncated Fourier feature map.
    This function encodes the truncated Fourier feature map as a symbolic expression in terms of the variables `xs`.
    The expression is constructed by summing the products of the weights, sigma_f, and the symbolic expressions
    for each row of the truncated Fourier feature map.
    It can then be used to bound the barrier function values with a formula like:

    ```py
    barrier_expression(xs, X_bounds, tffm, sigma_f, sol) >= 0
    ```

    to ensure that the barrier function is non-negative for all values within the bounds defined by `X_bounds`.

    Args:
        xs: list of variables (e.g., [x1, x2, ...])
        X_bounds: a RectSet representing the bounds of the input space
        tffm: TruncatedFourierFeatureMap object containing the Fourier feature map
        sigma_f: value of the variance
        sol: barrier certificate solution which uniquely identifies the barrier function

    Returns:
        A symbolic expression representing the barrier function.
    """
    # Encode the truncated Fourier feature map as a symbolic expression in terms of xs
    # Encode the truncated Fourier feature map as a symbolic expression in terms of xs
    sym_tffm = [1.0]
    for row in tffm.omega[1:]:
        sym_tffm.append(
            Cosine(
                sum(
                    o / (ub - lb) * (x - lb)
                    for o, x, lb, ub in zip(row, xs, X_bounds.lower_bound, X_bounds.upper_bound)
                )
            )
        )
        sym_tffm.append(
            Sine(
                sum(
                    o / (ub - lb) * (x - lb)
                    for o, x, lb, ub in zip(row, xs, X_bounds.lower_bound, X_bounds.upper_bound)
                )
            )
        )
    for i, (w, s) in enumerate(zip(tffm.weights, sol)):
        sym_tffm[i] *= w * sigma_f * s
    return sum(sym_tffm)


def verify_barrier_certificate(
    X_bounds: "RectSet",
    X_init: "Set",
    X_unsafe: "Set",
    sigma_f: float,
    eta: float,
    gamma: float,
    f_det: "Callable[[Real | float], Real | float]",
    c: float,
    estimator: "Estimator",
    tffm: "TruncatedFourierFeatureMap",
    sol: "NVector",
):
    """Use the dReal SMT solver to verify the barrier certificate for a given system.
    This function checks if the barrier certificate satisfies the conditions for safety and stability
    by constructing a set of constraints and checking their satisfiability.
    If a counterexample is found, it prints the counterexample point and the corresponding barrier values.

    Args:
        X_bounds: set of bounds for the state space
        X_init: set of initial states
        X_unsafe: set of unsafe states
        sigma_f: kernel bandwidth parameter
        eta: expected value of the barrier function at the initial state
        gamma: minimum value of the barrier function in the unsafe set
        f_det: transition function that maps each state to its corresponding next state
        c: maximum change in the barrier function value between successive states
        estimator: estimator object used to model the black-box system
        tffm: TruncatedFourierFeatureMap object containing the Fourier feature map
        sol: barrier certificate solution which uniquely identifies the barrier function

    Returns:
        True if the barrier certificate is verified, False otherwise.
    """
    # Create symbolic variables for the input dimensions
    xs = np.array([Real(f"x{i}") for i in range(X_bounds.dimension)])[np.newaxis, :]
    xsp = f_det(xs)
    xs = xs[0].tolist()  # Convert to a list for further processing
    xsp = xsp[0].tolist()  # Convert to a list for further processing
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
                Implies(build_set_constraint(xs, X_init), barrier <= eta + tolerance * eta),
                # Second condition
                Implies(build_set_constraint(xs, X_unsafe), barrier >= gamma - tolerance * gamma),
                # Third condition
                barrier_p - barrier <= c + tolerance * c,
            ),
        ),
    )
    res = CheckSatisfiability(constraints, 1e-8)
    if res is None:
        log.info("The barrier has been verified via dReal")
        return True

    log.error("Found counter example")
    model = {str(x): res[x].lb() for x in xs}
    log.error(f"Model: {model}")
    point = np.array([list(model.values())], dtype=np.float64)
    pointp = f_det(point)
    true_barrier = tffm(point) @ sol.T
    true_barrier_p = tffm(pointp) @ sol.T
    estimated_barrier = estimator(point) @ sol.T
    log.error(f"X: {point}, barrier value: {true_barrier}")
    log.error(f"Xp: {pointp}, barrier value: {true_barrier_p}")
    log.error(f"Xp: estimated, barrier value: {estimated_barrier}")
    log.error(f"Barrier at Xp {tffm(pointp)[0]}")
    log.error(f"Estimated barrier at Xp {estimator(point)[0]}")
    if (true_barrier < -tolerance).any():
        log.error("Violated barrier condition: B >= 0.")
    if point in X_unsafe and (true_barrier < gamma - tolerance * gamma).any():
        log.error("Violated barrier condition: B >= gamma.")
    if point in X_init and (true_barrier > eta + tolerance * eta).any():
        log.error("Violated barrier condition: B <= eta.")
    if (true_barrier_p - true_barrier > c + tolerance * c).any():
        log.error("Violated barrier condition: Bp - B <= c.")
    return False
