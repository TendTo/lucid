from typing import TYPE_CHECKING
from ._pylucid import RectSet, MultiSet, TruncatedFourierFeatureMap

try:
    from dreal import And, Or, sin as Sine, cos as Cosine
except ImportError as e:
    import sys

    print("dreal is not installed. Please install dreal to use this module.", file=sys.stderr)
    print("You can install it using 'pip install dreal'.", file=sys.stderr)
    raise e

if TYPE_CHECKING:
    import numpy as np


def set_constraint(xs: "list", X_set: "RectSet | MultiSet"):
    """
    Generate a set constraint for the given variables to ensure they lie within the bounds of the given set.

    Example:


    Args:
        xs: list of variables (e.g., [x1, x2, ...])
        X_set: a RectSet or MultiSet representing the bounds of the set
        (e.g., RectSet([0, 0], [1, 1]) or MultiSet(RectSet([0, 0], [1, 1]), RectSet([2, 2], [3, 3])))

    Raises:
        ValueError: if X_set is not a RectSet or MultiSet

    Returns:
        An expression representing the set constraint.
        For example, if X_set is RectSet([0, 0], [1, 1]), it returns (x1 >= 0) && (x1 <= 1) && (x2 >= 0) && (x2 <= 1).
    """
    if isinstance(X_set, RectSet):
        return And(*(b for i, x in enumerate(xs) for b in (x >= X_set.lower_bound[i], x <= X_set.upper_bound[i])))
    if isinstance(X_set, MultiSet):
        expr = None
        for rect in X_set:
            expr = Or(expr, set_constraint(xs, rect)) if expr is not None else set_constraint(xs, rect)
        return expr
    raise ValueError("X_set must be a RectSet or MultiSet.")


def barrier_expression(
    xs: "list",
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
