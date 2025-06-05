from ._pylucid import RectSet, MultiSet, FeatureMap, Estimator, LucidNotSupportedException, log_warn
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    log_warn("Could not import matplotlib. Make sure it is installed with 'pip install matplotlib'")
    raise e


def plot_set_1d(X_set: "RectSet | MultiSet", color: str, label: str = ""):
    """
    Plot the given set in 1D or 2D.

    Args:
        X_set: A RectSet or MultiSet representing the set to be plotted.
        color: The color to use for plotting the set.
    """
    if isinstance(X_set, RectSet):
        plt.plot((X_set.lower_bound, X_set.upper_bound), (0, 0), color=color, label=label)
    elif isinstance(X_set, MultiSet):
        for i, rect in enumerate(X_set):
            plt.plot((rect.lower_bound, rect.upper_bound), (0, 0), color=color, label=label if i == 0 else "")
    else:
        raise ValueError("X_set must be a RectSet or MultiSet.")


def plot_solution_1d(
    X_bounds: "RectSet",
    X_init: "RectSet" = None,
    X_unsafe: "RectSet" = None,
    feature_map: "FeatureMap" = None,
    sol: "np.typing.NDArray[np.float64]" = None,
    eta: float = None,
    gamma: float = None,
    estimator: "Estimator" = None,
    f: "callable" = None,
):
    plt.xlim(X_bounds.lower_bound, X_bounds.upper_bound)
    # Draw the initial and unsafe sets
    if X_init is not None:
        plot_set_1d(X_unsafe, "red", label="unsafe set")
    if X_unsafe is not None:
        plot_set_1d(X_init, "blue", label="initial set")

    if eta is not None:
        plt.plot(
            (X_bounds.lower_bound, X_bounds.upper_bound), (eta, eta), color="green", linestyle="dotted", label="eta"
        )
    if gamma is not None:
        plt.plot(
            (X_bounds.lower_bound, X_bounds.upper_bound), (gamma, gamma), color="red", linestyle="dotted", label="gamma"
        )

    if feature_map is not None and sol is not None:
        x_lattice = X_bounds.lattice(200, True)
        plt.plot(x_lattice, feature_map(x_lattice) @ sol.T, color="green", label="B(x)")
        if f is not None:
            plt.plot(x_lattice, feature_map(f(x_lattice.T).T) @ sol.T, color="black", label="B(xp)")
        if estimator is not None:
            plt.plot(
                x_lattice,
                estimator(x_lattice) @ sol.T,  # TODO(tend): Should this be regression(x_lattice, feature_map) @ sol.T?
                color="purple",
                label="B(xp) via approx. regression",
            )
        x_lattice_grid = X_bounds.lattice(feature_map.num_frequencies * 4, True)
        plt.scatter(x_lattice_grid, feature_map(x_lattice_grid) @ sol.T, color="green", label="B(x) (lattice)")
        if f is not None:
            plt.scatter(
                x_lattice_grid, feature_map(f(x_lattice_grid.T).T) @ sol.T, color="black", label="B(xp) (lattice)"
            )
        if estimator is not None:
            plt.scatter(
                x_lattice_grid,
                estimator(x_lattice_grid)
                @ sol.T,  # TODO(tend): Should this be regression(x_lattice, feature_map) @ sol.T?
                color="purple",
                label="B(xp) via approx. regression (lattice)",
            )

    plt.title("Barrier certificate")
    plt.xlabel("State space")
    plt.legend()
    plt.show()


def plot_solution_2d(
    X_bounds: "RectSet",
    X_init: "RectSet" = None,
    X_unsafe: "RectSet" = None,
    feature_map: "FeatureMap" = None,
    sol: "np.typing.NDArray[np.float64]" = None,
    eta: float = None,
    gamma: float = None,
    estimator: "Estimator" = None,
    f: "callable" = None,
):
    raise LucidNotSupportedException("2D plotting is not yet implemented. Please use 1D plotting instead.")


def plot_solution(
    X_bounds: "RectSet",
    X_init: "RectSet" = None,
    X_unsafe: "RectSet" = None,
    feature_map: "FeatureMap" = None,
    sol: "np.typing.NDArray[np.float64]" = None,
    eta: float = None,
    gamma: float = None,
    estimator: "Estimator" = None,
    f: "callable" = None,
):
    if X_bounds.dimension == 1:
        plot_solution_1d(X_bounds, X_init, X_unsafe, feature_map, sol, eta, gamma, estimator, f)
        return
    if X_bounds.dimension == 2:
        plot_solution_2d(X_bounds, X_init, X_unsafe, feature_map, sol, eta, gamma, estimator, f)
        return
    raise LucidNotSupportedException(
        f"Plotting is not supported for {X_bounds.dimension}-dimensional sets. Only 1D and 2D are supported."
    )
