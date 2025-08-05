import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pylucid import *

if TYPE_CHECKING:
    from typing import Callable
    from pylucid._pylucid import NMatrix, NVector

try:
    from pylucid.dreal import verify_barrier_certificate
except ImportError:

    def verify_barrier_certificate(*args, **kwargs):
        pass


import matplotlib.pyplot as plt

from pylucid import random

CM = 1 / 2.54  # centimeters in inches
plt.rcParams["font.family"] = ["Times New Roman"]


def plot_solution_matplotlib(
    args: "argparse.Namespace",
    X_bounds: "RectSet",
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    feature_map: "FeatureMap | None" = None,
    sol: "NVector | None" = None,
    eta: "float | None" = None,
    gamma: "float | None" = None,
    estimator: "Estimator | None" = None,
    f: "Callable[[NMatrix], NMatrix] | None" = None,
    c: "float" = 0.0,
    num_samples: "int | None" = None,
    show: bool = True,
) -> plt.Figure:
    assert X_bounds.dimension > 0, "X_bounds must have a positive dimension."
    plot_solution_fun = (plot_solution_1d_matplotlib, plot_solution_2d_matplotlib)
    if X_bounds.dimension <= len(plot_solution_fun):
        return plot_solution_fun[X_bounds.dimension - 1](
            args, X_bounds, X_init, X_unsafe, feature_map, sol, eta, gamma, estimator, f, c, num_samples, show
        )
    raise exception.LucidNotSupportedException(
        f"Plotting is not supported for {X_bounds.dimension}-dimensional sets. Only 1D and 2D are supported."
    )


def plot_solution_1d_matplotlib(
    args: "argparse.Namespace",
    X_bounds: "RectSet",
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    feature_map: "FeatureMap | None" = None,
    sol: "NVector | None" = None,
    eta: "float | None" = None,
    gamma: "float | None" = None,
    estimator: "Estimator | None" = None,
    f: "callable | None" = None,
    c: float = 0.0,
    num_samples: "int | None" = None,
    show: bool = True,
):
    """
    Alternative to plot_solution_1d using matplotlib instead of plotly.
    """

    # use an interactive backend if available

    fig, ax = plt.subplots()
    ax.set_xlim([X_bounds.lower_bound[0], X_bounds.upper_bound[0]])
    # Scale the y-axis to fit the plot to make the plot shorter
    # No set_ylim is set to allow the plot to scale automatically
    ax.set_aspect(aspect=0.7)

    # Draw the initial and unsafe sets
    def plot_rect_1d(s, color, label=None):
        if isinstance(s, RectSet):
            ax.plot([s.lower_bound[0], s.upper_bound[0]], [-0.005, -0.005], color=color, linewidth=3, label=label)
        elif isinstance(s, MultiSet):
            for i, rect in enumerate(s):
                ax.plot(
                    [rect.lower_bound[0], rect.upper_bound[0]],
                    [-0.005, -0.005],
                    color=color,
                    linewidth=3,
                    label=label if i == 0 else None,
                )

    if X_init is not None:
        plot_rect_1d(X_init, "blue", label=r"$X_0$")
    if X_unsafe is not None:
        plot_rect_1d(X_unsafe, "red", label=r"$X_U$")

    if eta is not None:
        ax.axhline(eta, linestyle="dotted", color="green", label=r"$\eta$")
    if gamma is not None:
        ax.axhline(gamma, linestyle="dotted", color="red", label=r"$\gamma$")

    if feature_map is not None and sol is not None:
        x_lattice = X_bounds.lattice(num_samples or 200, True)
        values = feature_map(x_lattice) @ sol.T
        ax.plot(x_lattice.flatten(), values.flatten(), color="green", label=r"$B(x)$")
        # ax.fill_between(x_lattice.flatten(), values, (values + c + 1e-8).flatten(), alpha=0.3, color="lightgreen", label="Barrier region")

        if f is not None and args.plot_bxp:
            f_values = feature_map(f(x_lattice)) @ sol.T
            ax.plot(x_lattice.flatten(), f_values.flatten(), color="black", label=r"$B(x_+)$")

        if estimator is not None and args.plot_bxe:
            est_values = estimator(x_lattice) @ sol.T
            ax.plot(x_lattice.flatten(), est_values.flatten(), color="purple", label=r"$B(x_p)$ est.")

        # Lattice points
        # x_lattice_grid = X_bounds.lattice(num_samples or (feature_map.num_frequencies * 4), True)
        # lattice_values = feature_map(x_lattice_grid) @ sol.T
        # ax.scatter(x_lattice_grid.flatten(), lattice_values.flatten(), color="green", s=10, label="B(x) (lattice)")

        # if f is not None:
        #     f_lattice_values = feature_map(f(x_lattice_grid)) @ sol.T
        #     ax.scatter(x_lattice_grid.flatten(), f_lattice_values.flatten(), color="black", s=10, label="B(xp) (lattice)")

        # if estimator is not None:
        #     est_lattice_values = estimator(x_lattice_grid) @ sol.T
        #     ax.scatter(x_lattice_grid.flatten(), est_lattice_values.flatten(), color="purple", s=10, label="B(xp) est. (lattice)")

    # Lower the legend to the bottom
    # Slighly increase the legend font size
    ax.legend(loc="center", ncol=2, fontsize=16, frameon=True)
    fig.tight_layout()
    fig.savefig(f"benchmarks/integration/{args.experiment.lower()}.pgf", bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    return fig, ax


def plot_solution_2d_matplotlib(
    args: "argparse.Namespace",
    X_bounds: "RectSet",
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    feature_map: "FeatureMap | None" = None,
    sol: "NVector | None" = None,
    eta: "float | None" = None,
    gamma: "float | None" = None,
    estimator: "Estimator | None" = None,
    f: "callable | None" = None,
    c: float = 0.0,
    num_samples: "int | None" = None,
    show: bool = True,
):
    """
    Alternative to plot_solution_2d using matplotlib instead of plotly.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Set plot bounds
    ax.set_xlim([X_bounds.lower_bound[0], X_bounds.upper_bound[0]])
    ax.set_ylim([X_bounds.lower_bound[1], X_bounds.upper_bound[1]])
    ax.set_zlim([0, gamma + 1 if gamma is not None else 1])
    # scale the z-axis to fit the plot to make the plot shorter
    ax.set_box_aspect([1, 1, 0.5])  # Aspect ratio for x, y, z axes

    # Helper function to plot sets in 2D on the z=0 plane
    def plot_rect_2d_matplotlib(s, color, label=None, alpha=0.3):
        if isinstance(s, RectSet):
            # Draw rectangle outline on z=0 plane
            x = [s.lower_bound[0], s.upper_bound[0], s.upper_bound[0], s.lower_bound[0], s.lower_bound[0]]
            y = [s.lower_bound[1], s.lower_bound[1], s.upper_bound[1], s.upper_bound[1], s.lower_bound[1]]
            z = [0, 0, 0, 0, 0]
            ax.plot(x, y, z, color=color, linewidth=3, label=label)

            # Fill the rectangle on z=0 plane for better visibility
            xx = [s.lower_bound[0], s.upper_bound[0], s.upper_bound[0], s.lower_bound[0]]
            yy = [s.lower_bound[1], s.lower_bound[1], s.upper_bound[1], s.upper_bound[1]]
            zz = [0, 0, 0, 0]
            ax.plot_trisurf(xx, yy, zz, color=color, alpha=alpha)

        elif isinstance(s, SphereSet):
            # Draw circle outline on z=0 plane
            theta = np.linspace(0, 2 * np.pi, 100)
            x = s.center[0] + s.radius * np.cos(theta)
            y = s.center[1] + s.radius * np.sin(theta)
            z = np.zeros_like(x)
            ax.plot(x, y, z, color=color, linewidth=3, label=label)
        elif isinstance(s, MultiSet):
            for i, rect in enumerate(s):
                plot_rect_2d_matplotlib(rect, color, label if i == 0 else None, alpha)

    # Draw the initial and unsafe sets as rectangles on the z=0 plane
    if X_init is not None:
        plot_rect_2d_matplotlib(X_init, "blue", label=r"$X_0$")
    if X_unsafe is not None:
        plot_rect_2d_matplotlib(X_unsafe, "red", label=r"$X_U$")

    # Plot the barrier certificate as a surface
    if feature_map is not None and sol is not None:
        x = np.linspace(X_bounds.lower_bound[0], X_bounds.upper_bound[0], num_samples or 25)
        y = np.linspace(X_bounds.lower_bound[1], X_bounds.upper_bound[1], num_samples or 25)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = feature_map(points) @ sol.T
        Z = Z.reshape(X.shape)

        # Plot main barrier surface
        surf = ax.plot_surface(
            X,
            Y,
            Z,
            cmap="viridis",
            alpha=0.7,
            label=r"$B(x)$",
            rstride=4,
            cstride=4,
        )
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

        plane_x = np.array([X_bounds.lower_bound[0], X_bounds.upper_bound[0]])
        plane_y = np.array([X_bounds.lower_bound[1], X_bounds.upper_bound[1]])
        plane_X, plane_Y = np.meshgrid(plane_x, plane_y)
        # Plot eta and gamma as planes if provided
        if eta is not None:
            eta_plane = ax.plot_surface(
                plane_X, plane_Y, np.full_like(plane_X, eta), color="green", alpha=0.2, label=r"$\eta$"
            )
            eta_plane._facecolors2d = eta_plane._facecolor3d
            eta_plane._edgecolors2d = eta_plane._edgecolor3d

        if gamma is not None:
            gamma_plane = ax.plot_surface(
                plane_X, plane_Y, np.full_like(plane_X, gamma), color="red", alpha=0.2, label=r"$\gamma$"
            )
            gamma_plane._facecolors2d = gamma_plane._facecolor3d
            gamma_plane._edgecolors2d = gamma_plane._edgecolor3d
            # Update z-axis limit if gamma is provided
            current_zlim = ax.get_zlim()
            ax.set_zlim([0, max(current_zlim[1], gamma + 1)])

        # Plot f(x) surface if provided
        if f is not None and args.plot_bxp:
            points_f = f(points)
            Zp = feature_map(points_f) @ sol.T
            Zp = Zp.reshape(X.shape)
            surf_f = ax.plot_surface(X, Y, Zp, color="black", alpha=0.3, label=r"$B(x_+)$")
            surf_f._facecolors2d = surf_f._facecolor3d
            surf_f._edgecolors2d = surf_f._edgecolor3d

        # Plot estimator surface if provided
        if estimator is not None and args.plot_bxe:
            Z_est = estimator(points) @ sol.T
            Z_est = Z_est.reshape(X.shape)
            surf_est = ax.plot_surface(X, Y, Z_est, color="purple", alpha=0.3, label=r"$B(x_p)$ est.")
            surf_est._facecolors2d = surf_est._facecolor3d
            surf_est._edgecolors2d = surf_est._edgecolor3d

    # Set labels and title
    ax.set_xlabel("State space x[0]")
    ax.set_ylabel("State space x[1]")
    ax.set_zlabel("Barrier value")

    # Move the legend on the rigt top
    # a bit more to the right
    ax.legend(loc="upper right", fontsize=16, frameon=True, bbox_to_anchor=(1.1, 0.9))

    # Save figure
    ax.view_init(args.elevation, args.azimuth, args.roll)
    fig.tight_layout()
    fig.savefig(f"benchmarks/integration/{args.experiment.lower()}.pgf", bbox_inches="tight", dpi=300)
    if show:
        plt.show()

    return fig, ax


def load_solution(file_path: "str | Path") -> "tuple[np.ndarray, float, float]":
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return np.array(data["data"], dtype=float).flatten(), data["eta"], data["c"]


def base_load_configuration(file_path: "str | Path") -> Configuration:
    config = Configuration.from_file(file_path)
    if config.seed >= 0:
        np.random.seed(config.seed)
        random.seed(config.seed)
    if len(config.x_samples) == 0:
        config.x_samples = config.X_bounds.sample(config.num_samples)
    if len(config.xp_samples) == 0:
        f = lambda x: config.system_dynamics(x) + (np.random.normal(scale=config.noise_scale))
        config.xp_samples = f(config.x_samples)
    return config


def load_configuration(file_path: "str | Path") -> Configuration:
    config = base_load_configuration(file_path)
    config.feature_map = config.feature_map(
        num_frequencies=config.num_frequencies,
        sigma_l=config.sigma_l,
        sigma_f=config.sigma_f,
        x_limits=config.X_bounds,
    )
    config.estimator = config.estimator(
        kernel=config.kernel(sigma_l=config.sigma_l, sigma_f=config.sigma_f),
        regularization_constant=config.lambda_,
    )
    config.estimator.consolidate(config.x_samples, config.feature_map(config.xp_samples))
    return config


def plot_contour_benchmarks(name: str, x: "np.ndarray", y: "np.ndarray", z: "np.ndarray"):
    fig = plt.figure(figsize=(11 * CM, 11 * CM))
    ax = fig.add_subplot(111)

    ax.tricontour(x, y, 1 - z, levels=len(np.unique(z)) // 2, linewidths=0.5, colors="k")
    cntr2 = ax.tricontourf(x, y, 1 - z, levels=len(np.unique(z)) // 2, cmap="RdBu")

    fig.colorbar(cntr2, ax=ax)
    ax.plot(x, y, "ko", ms=3)
    ax.set_xlabel("Number of frequencies", fontsize=11)
    ax.set_ylabel("Lattice size per dimension", fontsize=11)
    ax.set_xticks(np.linspace(min(x), max(x), max(x) - min(x) + 1, endpoint=True))
    fig.tight_layout()
    fig.savefig(f"benchmarks/integration/contour-{name.lower()}.pgf", bbox_inches="tight", dpi=300)
    plt.show()


def plot_solution(experiment_name: "str", args: "argparse.Namespace"):
    """
    Plot the solution using matplotlib.
    :param file_path: Path to the solution file.
    :param config: Configuration dictionary.
    :param solution: Solution data as a numpy array.
    """
    config = load_configuration(experiment_name + ".yaml")
    solution, eta, c = load_solution(experiment_name + ".sol.json")
    plot_solution_matplotlib(
        estimator=config.estimator,
        c=c,
        eta=eta,
        f=config.system_dynamics,
        feature_map=config.feature_map,
        gamma=config.gamma,
        num_samples=args.points,
        sol=solution,
        show=True,
        X_bounds=config.X_bounds,
        X_init=config.X_init,
        X_unsafe=config.X_unsafe,
        args=args,
    )
    if args.verify:
        print("Verifying barrier certificate...")
        verify_barrier_certificate(
            X_bounds=config.X_bounds,
            X_init=config.X_init,
            X_unsafe=config.X_unsafe,
            c=c,
            eta=eta,
            gamma=config.gamma,
            estimator=config.estimator,
            f_det=config.system_dynamics,
            sigma_f=config.sigma_f,
            sol=solution,
            tffm=config.feature_map,
        )


def validate_inputs_matplotlib(
    x_samples: "NMatrix | None",
    xp_samples: "NMatrix | None",
    X_bounds: "RectSet | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
):
    """Validate inputs for plotting functions."""
    dimensions = {s.dimension for s in (X_bounds, X_init, X_unsafe) if s is not None}
    assert len(dimensions) <= 1, "X_bounds, X_init, and X_unsafe must have the same dimension if provided."

    if x_samples is not None and xp_samples is not None:
        assert x_samples.shape == xp_samples.shape, "x_samples and xp_samples must have the same shape."
        if len(dimensions) == 1:
            assert x_samples.shape[1] == next(
                iter(dimensions)
            ), f"x_samples must have {next(iter(dimensions))} dimensions."


def plot_set_1d_matplotlib(X_set: "Set", color: str, label: str = "", ax=None):
    """Plot the given set in 1D using matplotlib."""
    if ax is None:
        fig, ax = plt.subplots()

    def plot_rect_1d_helper(s: "Set", color: str, label: str):
        if isinstance(s, RectSet):
            ax.plot([s.lower_bound[0], s.upper_bound[0]], [0, 0], color=color, linewidth=3, label=label)
        elif isinstance(s, SphereSet):
            ax.plot([s.center[0] - s.radius, s.center[0] + s.radius], [0, 0], color=color, linewidth=3, label=label)

    if isinstance(X_set, MultiSet):
        for i, subset in enumerate(X_set):
            plot_rect_1d_helper(subset, color, label if i == 0 else "")
    else:
        plot_rect_1d_helper(X_set, color, label)

    return ax


def plot_set_2d_matplotlib(X_set: "Set", color: str, label: str = "", ax=None):
    """Plot the given set in 2D using matplotlib."""
    if ax is None:
        fig, ax = plt.subplots()

    def plot_rect_2d_helper(s: "Set", color: str, label: str):
        if isinstance(s, RectSet):
            x = [s.lower_bound[0], s.upper_bound[0], s.upper_bound[0], s.lower_bound[0], s.lower_bound[0]]
            y = [s.lower_bound[1], s.lower_bound[1], s.upper_bound[1], s.upper_bound[1], s.lower_bound[1]]
            ax.plot(x, y, color=color, linewidth=2, label=label)
        elif isinstance(s, SphereSet):
            circle = plt.Circle((s.center[0], s.center[1]), s.radius, color=color, fill=False, linewidth=2, label=label)
            ax.add_patch(circle)

    if isinstance(X_set, MultiSet):
        for i, subset in enumerate(X_set):
            plot_rect_2d_helper(subset, color, label if i == 0 else "")
    else:
        plot_rect_2d_helper(X_set, color, label)

    return ax


def plot_function_1d_matplotlib(
    X_bounds: "RectSet",
    f: "Callable[[NMatrix], NMatrix]",
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    n: int = 100,
    show: bool = True,
) -> plt.Figure:
    """Plot a function f over the given samples in 1D using matplotlib."""
    assert X_bounds.dimension == 1, "plot_function is only supported for 1D functions."

    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate lattice points and compute function values
    x_samples = X_bounds.lattice(n, True).flatten()
    xp_samples = f(x_samples.reshape(-1, 1)).flatten()
    assert xp_samples.ndim == 1 or xp_samples.shape[1] == 1, "Function f must return a 1D array for 1D plotting."

    # Plot the vector field as arrows
    y_offset = 0.5  # Fixed y-coordinate for 1D visualization
    for i in range(0, len(x_samples), max(1, len(x_samples) // 20)):  # Subsample for cleaner visualization
        x_start, x_end = x_samples[i], xp_samples[i]
        ax.annotate(
            "", xy=(x_end, y_offset), xytext=(x_start, y_offset), arrowprops=dict(arrowstyle="->", color="blue", lw=1)
        )

    # Plot sets
    if X_init is not None:
        plot_set_1d_matplotlib(X_init, "blue", "$X_0$", ax)
    if X_unsafe is not None:
        plot_set_1d_matplotlib(X_unsafe, "red", "$X_U$", ax)

    ax.set_xlim([X_bounds.lower_bound[0], X_bounds.upper_bound[0]])
    ax.set_ylim([-0.5, 1.5])
    ax.set_xlabel("$x_1$", fontsize=16)
    if X_init is not None or X_unsafe is not None:
        ax.legend(fontsize=16)

    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_function_2d_matplotlib(
    X_bounds: "RectSet",
    f: "Callable[[NMatrix], NMatrix]",
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    n: int = 100,
    show: bool = True,
) -> plt.Figure:
    """Plot a function f over the given samples in 2D using matplotlib."""
    assert X_bounds.dimension == 2, "plot_function is only supported for 2D functions."

    fig, ax = plt.subplots(figsize=(13 * CM, 10 * CM))

    # Generate lattice points
    x_samples = X_bounds.lattice(n, True)
    X = x_samples[:, 0].reshape(n, n)
    Y = x_samples[:, 1].reshape(n, n)

    xp_samples = f(x_samples)

    assert xp_samples.ndim == 2 and xp_samples.shape[1] == 2, "Function f must return a 2D array for 2D plotting."

    Xp = xp_samples[:, 0].reshape(n, n)
    Yp = xp_samples[:, 1].reshape(n, n)

    # Compute direction vectors
    U = Xp - X
    V = Yp - Y

    # Create streamplot
    if True:
        ax.streamplot(X, Y, U, V, color="gray", density=1.5, linewidth=0.8, arrowsize=1.2)
        for noise in np.linspace(-0.01, 0.01, 10, endpoint=True):
            xp_noise = f(x_samples) + noise
            Xn = xp_noise[:, 0].reshape(n, n)
            Yn = xp_noise[:, 1].reshape(n, n)
            U_noise = Xn - X
            V_noise = Yn - Y
            # Use a lower alpha for noise to avoid clutter
            color = (0.5, 0.5, 0.5, 0.3)  # Gray with transparency
            ax.streamplot(X, Y, U_noise, V_noise, color=color, linewidth=0.3, density=1.5, arrowsize=0)
    else:
        import gstools as gs

        # the grid
        x = np.linspace(X_bounds.lower_bound[0], X_bounds.upper_bound[0], n)
        y = np.linspace(X_bounds.lower_bound[1], X_bounds.upper_bound[1], n)
        # a smooth Gaussian covariance model
        model = gs.Gaussian(dim=2, var=1, len_scale=0.01)
        srf = gs.SRF(model, generator="VectorField", seed=42)
        Xn, Yn = srf((X[0, :], Y[:, 0]), mesh_type="structured")
        print("Xn", Xn.shape, "Yn", Yn.shape)
        print(Xn[:5, :5], Yn[:5, :5])
        srf.plot()

        ax.streamplot(X, Y, U + Xn, V + Yn, color="gray", density=1.5, arrowsize=1.5)

    # Plot sets
    if X_init is not None:
        plot_set_2d_matplotlib(X_init, "blue", "$X_0$", ax)
    if X_unsafe is not None:
        plot_set_2d_matplotlib(X_unsafe, "red", "$X_U$", ax)

    ax.set_xlim([X_bounds.lower_bound[0], X_bounds.upper_bound[0]])
    ax.set_ylim([X_bounds.lower_bound[1], X_bounds.upper_bound[1]])
    ax.set_xlabel("$x_1$", fontsize=9)
    ax.set_ylabel("$x_2$", fontsize=9)
    ax.set_aspect("equal")
    if X_init is not None or X_unsafe is not None:
        ax.legend(fontsize=9)

    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_function_matplotlib(
    X_bounds: "RectSet",
    f: "Callable[[NMatrix], NMatrix]",
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    n: int = 100,
    show: bool = True,
) -> plt.Figure:
    """Plot a function f over the given samples using matplotlib."""
    plot_function_fun = (plot_function_1d_matplotlib, plot_function_2d_matplotlib)
    if X_bounds.dimension <= len(plot_function_fun):
        return plot_function_fun[X_bounds.dimension - 1](
            X_bounds=X_bounds,
            f=f,
            X_init=X_init,
            X_unsafe=X_unsafe,
            n=n,
            show=show,
        )
    raise exception.LucidNotSupportedException(
        f"Plotting is not supported for {X_bounds.dimension}-dimensional sets. Only 1D and 2D are supported."
    )


def plot_data_1d_matplotlib(
    X_bounds: "RectSet",
    x_samples: "NMatrix",
    xp_samples: "NMatrix",
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    show: bool = True,
) -> plt.Figure:
    """Plot data samples in 1D using matplotlib."""
    assert X_bounds.dimension == 1, "plot_data_1d is only supported for 1D data."

    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort data for better visualization
    idxs = x_samples.flatten().argsort()
    x_sub = x_samples.flatten()[idxs]
    xp_sub = xp_samples.flatten()[idxs]
    y_sub = np.linspace(0.4, 0.6, num=len(x_sub))  # y-coordinates are evenly spaced for 1D plot

    # Plot current and next state points
    ax.scatter(x_sub, y_sub, color="blue", s=20, alpha=0.7, label="Current state points")
    ax.scatter(xp_sub, y_sub, color="orange", s=20, alpha=0.7, label="Next state points")

    # Add arrows to show transitions (subsample for clarity)
    step = max(1, len(x_sub) // 50)
    for i in range(0, len(x_sub), step):
        ax.annotate(
            "",
            xy=(xp_sub[i], y_sub[i]),
            xytext=(x_sub[i], y_sub[i]),
            arrowprops=dict(arrowstyle="->", color="blue", lw=0.8, alpha=0.7),
        )

    # Plot sets
    if X_init is not None:
        plot_set_1d_matplotlib(X_init, "green", "$X_0$", ax)
    if X_unsafe is not None:
        plot_set_1d_matplotlib(X_unsafe, "red", "$X_U$", ax)

    ax.set_xlim([X_bounds.lower_bound[0], X_bounds.upper_bound[0]])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Input", fontsize=16)
    ax.set_ylabel("Output", fontsize=16)
    ax.legend(fontsize=16)

    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_data_2d_matplotlib(
    x_samples: "NMatrix",
    xp_samples: "NMatrix",
    X_bounds: "RectSet | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    show: bool = True,
) -> plt.Figure:
    """Plot data samples in 2D using matplotlib."""
    assert x_samples.ndim == 2 and x_samples.shape[1] == 2, "x_samples must be a 2D array with shape (n_samples, 2)."
    assert xp_samples.ndim == 2 and xp_samples.shape[1] == 2, "xp_samples must be a 2D array with shape (n_samples, 2)."

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot sets
    if X_init is not None:
        plot_set_2d_matplotlib(X_init, "green", "$X_0$", ax)
    if X_unsafe is not None:
        plot_set_2d_matplotlib(X_unsafe, "red", "$X_U$", ax)

    # Subsample for cleaner visualization
    step = max(1, len(x_samples) // 200)
    x_sub = x_samples[::step]
    xp_sub = xp_samples[::step]

    # Plot transitions as arrows
    for i in range(len(x_sub)):
        ax.annotate(
            "",
            xy=(xp_sub[i, 0], xp_sub[i, 1]),
            xytext=(x_sub[i, 0], x_sub[i, 1]),
            arrowprops=dict(arrowstyle="->", color="blue", lw=0.5, alpha=0.6),
        )

    # Plot sample points
    ax.scatter(x_sub[:, 0], x_sub[:, 1], color="blue", s=10, alpha=0.7, label="$x$")
    ax.scatter(xp_sub[:, 0], xp_sub[:, 1], color="orange", s=10, alpha=0.7, label="$x_+$")

    if X_bounds is not None:
        ax.set_xlim([X_bounds.lower_bound[0], X_bounds.upper_bound[0]])
        ax.set_ylim([X_bounds.lower_bound[1], X_bounds.upper_bound[1]])

    ax.set_xlabel("$x_1$", fontsize=16)
    ax.set_ylabel("$x_2$", fontsize=16)
    ax.legend(fontsize=16)

    fig.tight_layout()

    if show:
        plt.show()
    return fig


def plot_set_3d_matplotlib(X_set: "Set", color: str, label: str = "", ax=None):
    """Plot the given set in 3D using matplotlib."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    def plot_rect_3d_helper(s: "Set", color: str, label: str):
        if isinstance(s, RectSet):
            # Define the 8 vertices of the rectangular box
            x_min, y_min, z_min = s.lower_bound
            x_max, y_max, z_max = s.upper_bound

            # Define the 12 edges of the box
            edges = [
                # Bottom face edges
                ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
                ([x_max, x_max], [y_min, y_max], [z_min, z_min]),
                ([x_max, x_min], [y_max, y_max], [z_min, z_min]),
                ([x_min, x_min], [y_max, y_min], [z_min, z_min]),
                # Top face edges
                ([x_min, x_max], [y_min, y_min], [z_max, z_max]),
                ([x_max, x_max], [y_min, y_max], [z_max, z_max]),
                ([x_max, x_min], [y_max, y_max], [z_max, z_max]),
                ([x_min, x_min], [y_max, y_min], [z_max, z_max]),
                # Vertical edges
                ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
                ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
                ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
                ([x_min, x_min], [y_max, y_max], [z_min, z_max]),
            ]

            # Plot all edges
            for i, (x_coords, y_coords, z_coords) in enumerate(edges):
                ax.plot(x_coords, y_coords, z_coords, color=color, linewidth=2, label=label if i == 0 else "")

        elif isinstance(s, SphereSet):
            # Create a sphere using spherical coordinates
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x_sphere = s.center[0] + s.radius * np.outer(np.cos(u), np.sin(v))
            y_sphere = s.center[1] + s.radius * np.outer(np.sin(u), np.sin(v))
            z_sphere = s.center[2] + s.radius * np.outer(np.ones(np.size(u)), np.cos(v))

            # Plot wireframe sphere
            ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color=color, alpha=0.3, label=label)

    if isinstance(X_set, MultiSet):
        for i, subset in enumerate(X_set):
            plot_rect_3d_helper(subset, color, label if i == 0 else "")
    else:
        plot_rect_3d_helper(X_set, color, label)

    return ax


def plot_data_3d_matplotlib(
    x_samples: "NMatrix",
    xp_samples: "NMatrix",
    X_bounds: "RectSet | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    show: bool = True,
) -> plt.Figure:
    """Plot data samples in 3D using matplotlib."""
    assert x_samples.ndim == 2 and x_samples.shape[1] == 3, "x_samples must be a 2D array with shape (n_samples, 3)."
    assert xp_samples.ndim == 2 and xp_samples.shape[1] == 3, "xp_samples must be a 2D array with shape (n_samples, 3)."

    fig = plt.figure(figsize=(13 * CM, 13 * CM))
    ax = fig.add_subplot(111, projection="3d")

    # Subsample for cleaner visualization
    x_sub = x_samples
    xp_sub = xp_samples

    # Plot transitions as lines
    for i in range(len(x_sub)):
        ax.plot(
            [x_sub[i, 0], xp_sub[i, 0]],
            [x_sub[i, 1], xp_sub[i, 1]],
            [x_sub[i, 2], xp_sub[i, 2]],
            color="blue",
            linewidth=0.5,
            alpha=0.6,
        )

    # Plot sample points
    ax.scatter(x_sub[:, 0], x_sub[:, 1], x_sub[:, 2], color="blue", s=10, alpha=0.7, label="$x$")
    ax.scatter(xp_sub[:, 0], xp_sub[:, 1], xp_sub[:, 2], color="orange", s=10, alpha=0.7, label="$x_+$")

    if X_bounds is not None:
        ax.set_xlim([X_bounds.lower_bound[0], X_bounds.upper_bound[0]])
        ax.set_ylim([X_bounds.lower_bound[1], X_bounds.upper_bound[1]])
        ax.set_zlim([X_bounds.lower_bound[2], X_bounds.upper_bound[2]])

    if X_init is not None:
        plot_set_3d_matplotlib(X_init, "blue", "$X_0$", ax=ax)
    if X_unsafe is not None:
        plot_set_3d_matplotlib(X_unsafe, "red", "$X_U$", ax=ax)

    ax.set_xlabel("$x_1$", fontsize=11)
    ax.set_ylabel("$x_2$", fontsize=11)
    ax.set_zlabel("$x_3$", fontsize=11)
    # Move the legend on the right top and more to the right
    ax.legend(fontsize=11, loc="upper right", bbox_to_anchor=(1.1, 1.0))

    fig.tight_layout()
    ax.view_init(44, 154, 0)

    if show:
        plt.show()
    return fig


def plot_data_matplotlib(
    x_samples: "NMatrix",
    xp_samples: "NMatrix",
    X_bounds: "RectSet | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    show: bool = True,
) -> plt.Figure:
    """Plot data samples using matplotlib."""
    validate_inputs_matplotlib(
        x_samples=x_samples,
        xp_samples=xp_samples,
        X_bounds=X_bounds,
        X_init=X_init,
        X_unsafe=X_unsafe,
    )

    if X_bounds is not None:
        dimension = X_bounds.dimension
    else:
        dimension = x_samples.shape[1]

    plot_function_fun = (plot_data_1d_matplotlib, plot_data_2d_matplotlib, plot_data_3d_matplotlib)
    if dimension <= len(plot_function_fun):
        if dimension == 1:
            return plot_function_fun[dimension - 1](
                X_bounds=X_bounds,
                x_samples=x_samples,
                xp_samples=xp_samples,
                X_init=X_init,
                X_unsafe=X_unsafe,
                show=show,
            )
        else:
            return plot_function_fun[dimension - 1](
                x_samples=x_samples,
                xp_samples=xp_samples,
                X_bounds=X_bounds,
                X_init=X_init,
                X_unsafe=X_unsafe,
                show=show,
            )
    raise exception.LucidNotSupportedException(
        f"Plotting is not supported for {dimension}-dimensional sets. Only 1D, 2D and 3D are supported."
    )


# Convenience aliases to match the interface from plot.py
plot_function = plot_function_matplotlib
plot_data = plot_data_matplotlib


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot the solution of a barrier certificate.")
    parser.add_argument("experiment", type=str, help="The case study to plot.")
    parser.add_argument("-p", "--points", type=int, help="The number of points for the plot.", default=200)
    parser.add_argument("-e", "--elevation", type=float, help="The elevation angle for the plot.", default=30)
    parser.add_argument("-a", "--azimuth", type=float, help="The azimuth angle for the plot.", default=-15)
    parser.add_argument("-r", "--roll", type=float, help="The roll angle for the plot.", default=0)
    parser.add_argument("-v", "--verify", action="store_true", help="Verify the barrier certificate.")
    parser.add_argument("--plot_bxp", action="store_true", help="Plot the B(xp) surface.")
    parser.add_argument("--plot_bxe", action="store_true", help="Plot the B(xp) est. surface.")
    args = parser.parse_args()
    plot_solution(f"benchmarks/integration/{args.experiment}", args)
