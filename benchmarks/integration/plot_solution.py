import json
from pathlib import Path

import numpy as np

from pylucid import *

try:
    from pylucid.dreal import verify_barrier_certificate
except ImportError:

    def verify_barrier_certificate(*args, **kwargs):
        pass


import matplotlib.pyplot as plt

from pylucid import random


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
