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
    c: float = 0.0,
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
        x_lattice: np.ndarray = X_bounds.lattice(200, True)
        values = feature_map(x_lattice) @ sol.T
        plt.plot(x_lattice, values, color="green", label="B(x)")
        plt.fill_between(x_lattice.reshape(-1), values, values + c + 1e-8, color="lightgreen")
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


def plot_set_2d(X_set: "RectSet | MultiSet", color: str, label: str = ""):
    """
    Plot the given set in 2D.

    Args:
        X_set: A RectSet or MultiSet representing the set to be plotted.
        color: The color to use for plotting the set.
    """
    ax = plt.gca()

    def plot_rect_3d(rect, color, label=None):
        x = [rect.lower_bound[0], rect.upper_bound[0], rect.upper_bound[0], rect.lower_bound[0], rect.lower_bound[0]]
        y = [rect.lower_bound[1], rect.lower_bound[1], rect.upper_bound[1], rect.upper_bound[1], rect.lower_bound[1]]
        z = [0] * 5
        ax.plot(x, y, z, color=color, label=label)

    if isinstance(X_set, RectSet):
        plot_rect_3d(X_set, color, label)
    elif isinstance(X_set, MultiSet):
        for i, rect in enumerate(X_set):
            plot_rect_3d(rect, color, label if i == 0 else None)
    else:
        raise ValueError("X_set must be a RectSet or MultiSet.")


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
    c: float = 0.0,
):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(X_bounds.lower_bound[0], X_bounds.upper_bound[0])
    ax.set_ylim(X_bounds.lower_bound[1], X_bounds.upper_bound[1])
    ax.set_zlim(0)

    # Draw the initial and unsafe sets as rectangles on the z=0 plane
    plot_set_2d(X_init, "blue", label="initial set")
    plot_set_2d(X_unsafe, "red", label="unsafe set")

    # Plot the barrier certificate as a surface
    if feature_map is not None and sol is not None:
        x = np.linspace(X_bounds.lower_bound[0], X_bounds.upper_bound[0], 100)
        y = np.linspace(X_bounds.lower_bound[1], X_bounds.upper_bound[1], 100)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = feature_map(points) @ sol.T
        Z = Z.reshape(X.shape)
        surf = ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.7, linewidth=0, antialiased=True, label="B(x)")
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)

        # Plot eta and gamma as planes
        if eta is not None:
            ax.plot_surface(X, Y, np.full_like(X, eta), color="green", alpha=0.2, label="eta")
        if gamma is not None:
            ax.plot_surface(X, Y, np.full_like(X, gamma), color="red", alpha=0.2, label="gamma")

        # Plot f(x) and estimator if provided
        if f is not None:
            points_f = f(points.T).T
            Zp = feature_map(points_f) @ sol.T
            Zp = Zp.reshape(X.shape)
            ax.plot_surface(X, Y, Zp, color="black", alpha=0.3, label="B(xp)")
        if estimator is not None:
            Z_est = estimator(points) @ sol.T
            Z_est = Z_est.reshape(X.shape)
            ax.plot_surface(X, Y, Z_est, color="purple", alpha=0.3, label="B(xp) via approx. regression")

    ax.set_title("Barrier certificate")
    ax.set_xlabel("State space x[0]")
    ax.set_ylabel("State space x[1]")
    ax.set_zlabel("Barrier value")
    ax.legend()
    plt.show()


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
    c: float = 0.0,
):
    assert X_bounds.dimension > 0, "X_bounds must have a positive dimension."
    plot_solution_fun = (plot_solution_1d, plot_solution_2d)
    if X_bounds.dimension <= len(plot_solution_fun):
        return plot_solution_fun[X_bounds.dimension - 1](
            X_bounds, X_init, X_unsafe, feature_map, sol, eta, gamma, estimator, f, c
        )
    raise LucidNotSupportedException(
        f"Plotting is not supported for {X_bounds.dimension}-dimensional sets. Only 1D and 2D are supported."
    )
