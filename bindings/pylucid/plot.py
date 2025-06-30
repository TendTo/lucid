from typing import TYPE_CHECKING

import numpy as np

from ._pylucid import Estimator, FeatureMap, MultiSet, RectSet, Set, exception, log

if TYPE_CHECKING:
    from typing import Callable

    from mpl_toolkits.mplot3d.axes3d import Axes3D

    from ._pylucid import NMatrix, NVector

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    log.warn("Could not import matplotlib. Make sure it is installed with 'pip install matplotlib'")
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
    X_init: "RectSet | None" = None,
    X_unsafe: "RectSet | None" = None,
    feature_map: "FeatureMap | None" = None,
    sol: "NVector | None" = None,
    eta: "float | None" = None,
    gamma: "float | None" = None,
    estimator: "Estimator | None" = None,
    f: "callable | None" = None,
    c: float = 0.0,
    show: bool = True,
) -> "plt.Figure":
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
        x_lattice: np.ndarray = X_bounds.lattice(200, True)
        values = feature_map(x_lattice) @ sol.T
        plt.plot(x_lattice, values, color="green", label="B(x)")
        plt.fill_between(x_lattice.reshape(-1), values, values + c + 1e-8, color="lightgreen")
        if f is not None:
            plt.plot(x_lattice, feature_map(f(x_lattice)) @ sol.T, color="black", label="B(xp)")
        if estimator is not None:
            plt.plot(
                x_lattice,
                estimator(x_lattice) @ sol.T,  # TODO(tend): Should this be regression(x_lattice, feature_map) @ sol.T?
                color="purple",
                label="B(xp) via regression",
            )
        x_lattice_grid = X_bounds.lattice(feature_map.num_frequencies * 4, True)
        plt.scatter(x_lattice_grid, feature_map(x_lattice_grid) @ sol.T, color="green", label="B(x) (lattice)")
        if f is not None:
            plt.scatter(x_lattice_grid, feature_map(f(x_lattice_grid)) @ sol.T, color="black", label="B(xp) (lattice)")
        if estimator is not None:
            plt.scatter(
                x_lattice_grid,
                estimator(x_lattice_grid)
                @ sol.T,  # TODO(tend): Should this be regression(x_lattice, feature_map) @ sol.T?
                color="purple",
                label="B(xp) via regression (lattice)",
            )

    plt.title("Barrier certificate")
    plt.xlabel("State space")
    plt.legend()
    if show:
        plt.show()
    return plt.gcf()


def plot_set_2d(X_set: "RectSet | MultiSet", color: str, label: str = ""):
    """
    Plot the given set in 2D.

    Args:
        X_set: A RectSet or MultiSet representing the set to be plotted.
        color: The color to use for plotting the set.
    """
    ax = plt.gca()

    def plot_rect_2d(rect, color, label=None):
        x = [rect.lower_bound[0], rect.upper_bound[0], rect.upper_bound[0], rect.lower_bound[0], rect.lower_bound[0]]
        y = [rect.lower_bound[1], rect.lower_bound[1], rect.upper_bound[1], rect.upper_bound[1], rect.lower_bound[1]]
        ax.plot(x, y, 0, color=color, label=label)

    if isinstance(X_set, RectSet):
        plot_rect_2d(X_set, color, label)
    elif isinstance(X_set, MultiSet):
        for i, rect in enumerate(X_set):
            plot_rect_2d(rect, color, label if i == 0 else None)
    else:
        raise ValueError("X_set must be a RectSet or MultiSet.")


def plot_solution_2d(
    X_bounds: "RectSet",
    X_init: "RectSet | None" = None,
    X_unsafe: "RectSet | None" = None,
    feature_map: "FeatureMap | None" = None,
    sol: "NVector | None" = None,
    eta: "float | None" = None,
    gamma: "float | None" = None,
    estimator: "Estimator" = None,
    f: "Callable[[NMatrix], NMatrix] | None" = None,
    c: float = 0.0,
    show: bool = True,
) -> "plt.Figure":
    fig = plt.figure()
    ax: "Axes3D" = fig.add_subplot(111, projection="3d")
    ax.set_xlim(X_bounds.lower_bound[0], X_bounds.upper_bound[0])
    ax.set_ylim(X_bounds.lower_bound[1], X_bounds.upper_bound[1])
    ax.set_zlim(0)

    # Draw the initial and unsafe sets as rectangles on the z=0 plane
    plot_set_2d(X_init, "blue", label="initial set")
    plot_set_2d(X_unsafe, "red", label="unsafe set")

    # Plot the barrier certificate as a surface
    if feature_map is not None and sol is not None:
        x = np.linspace(X_bounds.lower_bound[0], X_bounds.upper_bound[0], 25)
        y = np.linspace(X_bounds.lower_bound[1], X_bounds.upper_bound[1], 25)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = feature_map(points) @ sol.T
        Z = Z.reshape(X.shape)
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, label="B(x)")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

        # Plot eta and gamma as planes
        if eta is not None:
            ax.plot_surface(X, Y, np.full_like(X, eta), color="green", alpha=0.2, rcount=2, ccount=2, label="eta")
        if gamma is not None:
            ax.plot_surface(X, Y, np.full_like(X, gamma), color="red", alpha=0.2, rcount=2, ccount=2, label="gamma")
            ax.set_zlim(0, gamma + 1)

        # Plot f(x) and estimator if provided
        if f is not None:
            points_f = f(points)
            Zp = feature_map(points_f) @ sol.T
            Zp = Zp.reshape(X.shape)
            ax.plot_surface(X, Y, Zp, color="black", alpha=0.3, label="B(xp)")
        if estimator is not None:
            Z_est = estimator(points) @ sol.T
            Z_est = Z_est.reshape(X.shape)
            ax.plot_surface(X, Y, Z_est, color="purple", alpha=0.3, label="B(xp) via regression")

    ax.set_title("Barrier certificate")
    ax.set_xlabel("State space x[0]")
    ax.set_ylabel("State space x[1]")
    ax.set_zlabel("Barrier value")
    ax.legend()
    if show:
        plt.show()
    return plt.gcf()


def plot_estimator_1d(
    estimator: "Estimator",
    x_samples: "NMatrix",
    xp_samples: "NMatrix",
    X_bounds: "RectSet | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    show: bool = True,
):
    """Plot the estimator's predictions against the true system dynamics in 2D.

    Args:
        estimator: The estimator to be evaluated
        x_samples: Sample points in the state space
        xp_samples: Sample points at the next time step
        X_init: Initial set of states
        X_unsafe: Unsafe set of states
        X_bounds: Bounds of the state space
        show: Whether to display the plot
    """
    if X_bounds is not None:
        plt.xlim(X_bounds.lower_bound[0], X_bounds.upper_bound[0])

    if X_init is not None:
        plot_set_1d(X_init, label="Initial Set", color="green")
    if X_unsafe is not None:
        plot_set_1d(X_unsafe, label="Unsafe Set", color="red")

    xp_pred = estimator.predict(x_samples)

    # Plot the true vs predicted next states
    plt.scatter(x_samples, xp_samples, label="Ground truth", color="blue", marker="o")
    plt.scatter(x_samples, xp_pred, label="Estimator prediction", color="orange", marker="x")
    plt.title("Estimator Predictions vs True Dynamics")
    plt.xlabel("State")
    plt.ylabel("Next State")
    plt.legend()
    if show:
        plt.show()
    return plt.gcf()


def plot_estimator_2d(
    estimator: "Estimator",
    x_samples: "NMatrix",
    xp_samples: "NMatrix",
    X_bounds: "RectSet | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    show: bool = True,
):
    """Plot the estimator's predictions against the true system dynamics in 2D.

    Args:
        estimator: The estimator to be evaluated
        x_samples: Sample points in the state space
        xp_samples: Sample points at the next time step
        X_init: Initial set of states
        X_unsafe: Unsafe set of states
        X_bounds: Bounds of the state space
        show: Whether to display the plot
    """
    fig = plt.figure()
    ax: "Axes3D" = fig.add_subplot(111, projection="3d")
    ax.set_xlim(X_bounds.lower_bound[0], X_bounds.upper_bound[0])
    ax.set_ylim(X_bounds.lower_bound[1], X_bounds.upper_bound[1])

    if X_init is not None:
        plot_set_2d(X_init, label="Initial Set", color="green")
    if X_unsafe is not None:
        plot_set_2d(X_unsafe, label="Unsafe Set", color="red")

    xp_pred = estimator.predict(x_samples)

    # Plot the true vs predicted next states
    ax.scatter(x_samples[:, 0], x_samples[:, 1], xp_samples[:, 0], label="Ground truth", color="blue", marker="o")
    ax.scatter(
        x_samples[:, 0], x_samples[:, 1], xp_pred[:, 0], label="Estimator prediction", color="orange", marker="x"
    )
    plt.title("Estimator Predictions vs True Dynamics")
    plt.xlabel("State x[0]")
    plt.ylabel("State x[1]")
    ax.set_zlabel("Next State")
    plt.legend()
    if show:
        plt.show()
    return plt.gcf()


def plot_feature_map(
    feature_map: "FeatureMap",
    x_samples: "NMatrix | Callable[[], NMatrix] | None" = None,
    X_bounds: "Set | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    num_plots: int = 10,
    N: int = 300,
    show: bool = True,
) -> "plt.Figure":
    """Plot the feature map of the system dynamics.

    Args:
        feature_map: The feature map to plot
        x_samples: Sample points in the state space
        xp_samples: Function to compute next state samples
        X_bounds: Bounds of the state space
        X_init: Initial state set
        X_unsafe: Unsafe state set
        N: Number of samples for plotting
        show: Whether to display the plot
    """
    assert feature_map is not None, "Feature map must be provided."
    assert N > 0, "N must be a positive integer."

    # If we are given a set, we get a lattice of points
    if callable(x_samples):  # If we are given a function, we call it to get the samples
        x_samples = x_samples()
    elif x_samples is None:
        x_samples = X_bounds.lattice(N, True)

    for _ in range(num_plots):
        B_x = feature_map(x_samples)
        sol = np.random.rand(B_x.shape[1]) * 20 - 10  # Random solution vector, from -10 to 10
        val = B_x @ sol.T
        plt.plot(x_samples, val)

    if X_bounds is not None:
        plt.xlim(X_bounds.lower_bound, X_bounds.upper_bound)
    if X_init is not None:
        plot_set_1d(X_init, "blue", label="Initial Set")
    if X_unsafe is not None:
        plot_set_1d(X_unsafe, "red", label="Unsafe Set")
    plt.title("Feature Map Visualization")
    plt.xlabel("State space")
    plt.ylabel("Feature Map Value")
    plt.legend()
    if show:
        plt.show()
    return plt.gcf()


def plot_estimator(
    estimator: "Estimator",
    x_samples: "NMatrix | Callable[[], NMatrix] | None",
    xp_samples: "NMatrix | Callable[[NMatrix], NMatrix]",
    X_bounds: "RectSet | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    N: int = 300,
    show: bool = True,
) -> "plt.Figure":
    """Plot the estimator's predictions against the true system dynamics.

    Args:
        estimator: The estimator to be evaluated
        x_samples: Sample points in the state space, a function that returns them or the set from which to sample them
        xp_samples: Sample points at the next time step or a function that returns them given x_samples
        X_init: Initial set of states
        X_unsafe: Unsafe set of states
        N: Number of samples to generate if x_samples is not provided
    """

    assert estimator is not None, "Estimator must be provided."
    assert xp_samples is not None, "xp_samples must be provided."
    assert N > 0, "N must be a positive integer."

    # If we are given a set, we get a lattice of points
    if callable(x_samples):  # If we are given a function, we call it to get the samples
        x_samples = x_samples()
    elif x_samples is None:
        x_samples = X_bounds.lattice(N, True)

    if callable(xp_samples):  # If xp_samples is a function, evaluate it on the x_samples
        xp_samples = xp_samples(x_samples)

    assert x_samples.ndim == 2, "to avoid ambiguity, x_samples must be a 2D array (num_samples x num_dimensions)."
    assert xp_samples.ndim == 2, "to avoid ambiguity, xp_samples must be a 2D array (num_samples x num_dimensions)."

    plot_estimator_fun = (plot_estimator_1d, plot_estimator_2d)
    if x_samples.shape[1] <= len(plot_estimator_fun) and xp_samples.shape[1] == 1:
        return plot_estimator_fun[x_samples.shape[1] - 1](
            estimator, x_samples, xp_samples, X_bounds, X_init, X_unsafe, show
        )
    raise exception.LucidNotSupportedException(
        f"Plotting is not supported for {x_samples.shape[1]} => {xp_samples.shape[1]}-dimensional sets. "
        f"Only (1D or 2D) => 1D are supported."
    )


def plot_solution(
    X_bounds: "RectSet",
    X_init: "RectSet | None" = None,
    X_unsafe: "RectSet | None" = None,
    feature_map: "FeatureMap | None" = None,
    sol: "NVector | None" = None,
    eta: "float | None" = None,
    gamma: "float | None" = None,
    estimator: "Estimator | None" = None,
    f: "Callable[[NMatrix], NMatrix] | None" = None,
    c: "float" = 0.0,
    show: bool = True,
) -> "plt.Figure":
    assert X_bounds.dimension > 0, "X_bounds must have a positive dimension."
    plot_solution_fun = (plot_solution_1d, plot_solution_2d)
    if X_bounds.dimension <= len(plot_solution_fun):
        return plot_solution_fun[X_bounds.dimension - 1](
            X_bounds, X_init, X_unsafe, feature_map, sol, eta, gamma, estimator, f, c, show
        )
    raise exception.LucidNotSupportedException(
        f"Plotting is not supported for {X_bounds.dimension}-dimensional sets. Only 1D and 2D are supported."
    )


def plot_function_1d(
    X_bounds: "RectSet",
    f: "Callable[[NMatrix], NMatrix]",
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    n: int = 100,
    show: bool = True,
) -> "plt.Figure":
    """Plot a function f over the given samples in 1D.

    Args:
        X_bounds: Bounds of the state space
        f: Function to be plotted
        X_init: Initial set of states
        X_unsafe: Unsafe set of states
        n: Number of lattice points to consider for plotting
        show: Whether to display the plot
    """
    assert X_bounds.dimension == 1, "plot_function is only supported for 1D functions."
    if X_init is not None:
        plot_set_1d(X_init, "blue", label="Initial Set")
    if X_unsafe is not None:
        plot_set_1d(X_unsafe, "red", label="Unsafe Set")

    x_samples = X_bounds.lattice(n, True).flatten().reshape(-1, 1)  # Ensure x_samples is a 2D array with shape (n, 1)
    y = np.linspace(0, 100, n).flatten()
    y_samples = f(x_samples)
    assert y_samples.ndim == 1 or y_samples.shape[1] == 1, "Function f must return a 1D array for 1D plotting."

    u = np.repeat((y_samples - x_samples).reshape(1, -1), n, axis=0)
    v = np.zeros((n, n))  # Assuming a 1D function, v is zero

    plt.streamplot(x_samples, y, u, v, color="blue", density=1 / 5, linewidth=u)
    plt.title("Function Plot")
    plt.xlabel("Input")
    plt.ylabel("Output")
    if show:
        plt.show()
    return plt.gcf()


def plot_function_2d(
    X_bounds: "RectSet",
    f: "Callable[[NMatrix], NMatrix]",
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    n: int = 100,
    show: bool = True,
) -> "plt.Figure":
    """Plot a function f over the given samples in 2D.

    Args:
        X_bounds: Bounds of the state space
        f: Function to be plotted
        X_init: Initial set of states
        X_unsafe: Unsafe set of states
        n: Number of lattice points to consider for plotting
        show: Whether to display the plot
    """
    assert X_bounds.dimension == 2, "plot_function is only supported for 2D functions."
    if X_init is not None:
        plot_set_2d(X_init, "blue", label="Initial Set")
    if X_unsafe is not None:
        plot_set_2d(X_unsafe, "red", label="Unsafe Set")

    x_samples = X_bounds.lattice(n, True)
    X = x_samples[:, 0].reshape(n, n)
    Y = x_samples[:, 1].reshape(n, n)

    xp_samples = f(x_samples)

    assert xp_samples.ndim == 2 and xp_samples.shape[1] == 2, "Function f must return a 2D array for 2D plotting."

    Xp = xp_samples[:, 0].reshape(n, n)
    Yp = xp_samples[:, 1].reshape(n, n)

    u = Xp - X
    v = Yp - Y

    speed = np.sqrt(u**2 + v**2)
    speed[speed == 0] = 1

    print(f"Speed: {speed.max()}")
    plt.streamplot(X, Y, u, v, color="blue", linewidth=5 * speed / speed.max())
    plt.title("Function Plot")
    plt.xlabel("Input Dimension 1")
    plt.ylabel("Input Dimension 2")
    if show:
        plt.show()
    return plt.gcf()


def plot_function(
    X_bounds: "RectSet",
    f: "Callable[[NMatrix], NMatrix]",
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    show: bool = True,
):
    """Plot a function f over the given samples."""
    plot_function_fun = (plot_function_1d, plot_function_2d)  # Add more functions for higher dimensions if needed
    if X_bounds.dimension <= len(plot_function_fun):
        return plot_function_fun[X_bounds.dimension - 1](X_bounds, f, X_init, X_unsafe, show=show)
    raise exception.LucidNotSupportedException(
        f"Plotting is not supported for {X_bounds.dimension}-dimensional sets. Only 1D and 2D are supported."
    )
