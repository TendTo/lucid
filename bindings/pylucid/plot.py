from typing import TYPE_CHECKING

import numpy as np

from ._pylucid import Estimator, FeatureMap, MultiSet, RectSet, Set, exception, log
from .util import assert_or_raise

if TYPE_CHECKING:
    from typing import Callable

    from ._pylucid import NMatrix, NVector

try:
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
except ImportError as e:
    log.warn("Could not import plotly. Make sure it is installed with 'pip install plotly'")
    raise e


def plot_set(
    plt_fun: "Callable[[Set, str, str, go.Figure], None]",
    x_set: "RectSet | MultiSet",
    color: str,
    label: str = "",
    fig: go.Figure = None,
) -> go.Figure:
    if fig is None:
        fig = go.Figure()
    if isinstance(x_set, RectSet):
        plt_fun(x_set, color, label, fig)
    elif isinstance(x_set, MultiSet):
        for i, rect in enumerate(x_set):
            plt_fun(rect, color, label if i == 0 else "", fig)
    else:
        raise ValueError("X_set must be a RectSet or MultiSet.")

    return fig


def plot_set_1d(X_set: "RectSet | MultiSet", color: str, label: str = "", fig: go.Figure = None) -> go.Figure:
    """
    Plot the given set in 1D.

    Args:
        X_set: A RectSet or MultiSet representing the set to be plotted.
        color: The color to use for plotting the set.
        label: Label for the set.
        fig: Existing figure to add to.
    """

    def plot_rect_1d(rect: RectSet, color: str, label: str, fig: go.Figure):
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=rect.lower_bound[0],
            y0=0,
            x1=rect.upper_bound[0],
            y1=0,
            line=dict(color=color, width=3),
            name=label,
            showlegend=bool(label),
        )

    return plot_set(plot_rect_1d, X_set, color, label, fig)


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
    num_samples: "int | None" = None,
    show: bool = True,
) -> go.Figure:
    fig = go.Figure()

    # Set plot bounds
    fig.update_xaxes(range=[X_bounds.lower_bound[0], X_bounds.upper_bound[0]])

    # Draw the initial and unsafe sets
    if X_unsafe is not None:
        fig = plot_set_1d(X_unsafe, "red", label="unsafe set", fig=fig)
    if X_init is not None:
        fig = plot_set_1d(X_init, "blue", label="initial set", fig=fig)

    if eta is not None:
        fig.add_hline(
            y=eta, line_dash="dot", line_color="green", annotation_text="eta", annotation_position="top right"
        )
    if gamma is not None:
        fig.add_hline(
            y=gamma, line_dash="dot", line_color="red", annotation_text="gamma", annotation_position="top right"
        )

    if feature_map is not None and sol is not None:
        x_lattice: np.ndarray = X_bounds.lattice(num_samples or 200, True)
        values = feature_map(x_lattice) @ sol.T

        # Plot B(x)
        fig.add_trace(
            go.Scatter(x=x_lattice.flatten(), y=values.flatten(), mode="lines", line=dict(color="green"), name="B(x)")
        )

        # Fill area for barrier
        fig.add_trace(
            go.Scatter(
                x=x_lattice.flatten(),
                y=(values + c + 1e-8).flatten(),
                fill="tonexty",
                fillcolor="rgba(144, 238, 144, 0.3)",
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                name="Barrier region",
            )
        )

        if f is not None:
            f_values = feature_map(f(x_lattice)) @ sol.T
            fig.add_trace(
                go.Scatter(
                    x=x_lattice.flatten(), y=f_values.flatten(), mode="lines", line=dict(color="black"), name="B(xp)"
                )
            )

        if estimator is not None:
            est_values = estimator(x_lattice) @ sol.T
            fig.add_trace(
                go.Scatter(
                    x=x_lattice.flatten(),
                    y=est_values.flatten(),
                    mode="lines",
                    line=dict(color="purple"),
                    name="B(xp) via regression",
                )
            )

        # Lattice points
        x_lattice_grid = X_bounds.lattice(num_samples or (feature_map.num_frequencies * 4), True)
        lattice_values = feature_map(x_lattice_grid) @ sol.T
        fig.add_trace(
            go.Scatter(
                x=x_lattice_grid.flatten(),
                y=lattice_values.flatten(),
                mode="markers",
                marker=dict(color="green"),
                name="B(x) (lattice)",
            )
        )

        if f is not None:
            f_lattice_values = feature_map(f(x_lattice_grid)) @ sol.T
            fig.add_trace(
                go.Scatter(
                    x=x_lattice_grid.flatten(),
                    y=f_lattice_values.flatten(),
                    mode="markers",
                    marker=dict(color="black"),
                    name="B(xp) (lattice)",
                )
            )

        if estimator is not None:
            est_lattice_values = estimator(x_lattice_grid) @ sol.T
            fig.add_trace(
                go.Scatter(
                    x=x_lattice_grid.flatten(),
                    y=est_lattice_values.flatten(),
                    mode="markers",
                    marker=dict(color="purple"),
                    name="B(xp) via regression (lattice)",
                )
            )

    fig.update_layout(title="Barrier certificate", xaxis_title="State space", showlegend=True)

    if show:
        fig.show()
    return fig


def plot_set_2d(X_set: "RectSet | MultiSet", color: str, label: str = "", fig: go.Figure = None):
    """
    Plot the given set in 2D.

    Args:
        X_set: A RectSet or MultiSet representing the set to be plotted.
        color: The color to use for plotting the set.
        label: Label for the set.
        fig: Existing figure to add to.
    """

    def plot_rect_2d(rect: RectSet, color: str, label: str, fig: go.Figure):
        x = [rect.lower_bound[0], rect.upper_bound[0], rect.upper_bound[0], rect.lower_bound[0], rect.lower_bound[0]]
        y = [rect.lower_bound[1], rect.lower_bound[1], rect.upper_bound[1], rect.upper_bound[1], rect.lower_bound[1]]
        z = [0, 0, 0, 0, 0]

        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z, mode="lines", line=dict(color=color, width=5), name=label, showlegend=bool(label)
            )
        )

    return plot_set(plot_rect_2d, X_set, color, label, fig)


def plot_set_2d_plane(X_set: "RectSet | MultiSet", color: str, label: str = "", fig: go.Figure = None):
    """
    Plot the given set in 2D.

    Args:
        X_set: A RectSet or MultiSet representing the set to be plotted.
        color: The color to use for plotting the set.
        label: Label for the set.
        fig: Existing figure to add to.
    """

    def plot_rect_2d(rect: RectSet, color: str, label: str, fig: go.Figure):
        fig.add_shape(
            type="rect",
            x0=rect.lower_bound[0],
            y0=rect.lower_bound[1],
            x1=rect.upper_bound[0],
            y1=rect.upper_bound[1],
            line=dict(color=color),
            name=label,
            showlegend=bool(label),
        )

    return plot_set(plot_rect_2d, X_set, color, label, fig)


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
    num_samples: "int | None" = None,
    show: bool = True,
) -> go.Figure:
    fig = go.Figure()

    # Set plot bounds
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[X_bounds.lower_bound[0], X_bounds.upper_bound[0]]),
            yaxis=dict(range=[X_bounds.lower_bound[1], X_bounds.upper_bound[1]]),
            zaxis=dict(range=[0, None]),
        )
    )

    # Draw the initial and unsafe sets as rectangles on the z=0 plane
    if X_init is not None:
        fig = plot_set_2d(X_init, "blue", label="initial set", fig=fig)
    if X_unsafe is not None:
        fig = plot_set_2d(X_unsafe, "red", label="unsafe set", fig=fig)

    # Plot the barrier certificate as a surface
    if feature_map is not None and sol is not None:
        x = np.linspace(X_bounds.lower_bound[0], X_bounds.upper_bound[0], num_samples or 25)
        y = np.linspace(X_bounds.lower_bound[1], X_bounds.upper_bound[1], num_samples or 25)
        X, Y = np.meshgrid(x, y)
        points = np.stack([X.ravel(), Y.ravel()], axis=1)
        Z = feature_map(points) @ sol.T
        Z = Z.reshape(X.shape)

        fig.add_trace(
            go.Surface(x=X, y=Y, z=Z, colorscale="Viridis", opacity=0.7, name="B(x)", showscale=False, showlegend=True)
        )

        # Plot eta and gamma as planes
        if eta is not None:
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=np.full_like(X, eta),
                    colorscale=[[0, "green"], [1, "green"]],
                    opacity=0.2,
                    name="eta",
                    showscale=False,
                    showlegend=True,
                )
            )
        if gamma is not None:
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=np.full_like(X, gamma),
                    colorscale=[[0, "red"], [1, "red"]],
                    opacity=0.2,
                    name="gamma",
                    showscale=False,
                    showlegend=True,
                )
            )
            fig.update_layout(scene=dict(zaxis=dict(range=[0, gamma + 1])))

        # Plot f(x) and estimator if provided
        if f is not None:
            points_f = f(points)
            Zp = feature_map(points_f) @ sol.T
            Zp = Zp.reshape(X.shape)
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Zp,
                    colorscale=[[0, "black"], [1, "black"]],
                    opacity=0.3,
                    name="B(xp)",
                    showscale=False,
                    showlegend=True,
                )
            )

        if estimator is not None:
            Z_est = estimator(points) @ sol.T
            Z_est = Z_est.reshape(X.shape)
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Z_est,
                    colorscale=[[0, "purple"], [1, "purple"]],
                    opacity=0.3,
                    name="B(xp) via regression",
                    showscale=False,
                    showlegend=True,
                )
            )

    fig.update_layout(
        title="Barrier certificate",
        scene=dict(xaxis_title="State space x[0]", yaxis_title="State space x[1]", zaxis_title="Barrier value"),
    )

    if show:
        fig.show()
    return fig


def plot_estimator_1d(
    estimator: "Estimator",
    x_samples: "NMatrix",
    xp_samples: "NMatrix",
    X_bounds: "RectSet | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    show: bool = True,
) -> go.Figure:
    """Plot the estimator's predictions against the true system dynamics in 1D."""
    fig = go.Figure()

    if X_bounds is not None:
        fig.update_xaxes(range=[X_bounds.lower_bound[0], X_bounds.upper_bound[0]])

    if X_init is not None:
        fig = plot_set_1d(X_init, color="green", label="Initial Set", fig=fig)
    if X_unsafe is not None:
        fig = plot_set_1d(X_unsafe, color="red", label="Unsafe Set", fig=fig)

    xp_pred = estimator.predict(x_samples)

    # Plot the true vs predicted next states
    fig.add_trace(
        go.Scatter(
            x=x_samples.flatten(),
            y=xp_samples.flatten(),
            mode="markers",
            marker=dict(color="blue", symbol="circle"),
            name="Ground truth",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=x_samples.flatten(),
            y=xp_pred.flatten(),
            mode="markers",
            marker=dict(color="orange", symbol="x"),
            name="Estimator prediction",
        )
    )

    fig.update_layout(title="Estimator Predictions vs True Dynamics", xaxis_title="State", yaxis_title="Next State")

    if show:
        fig.show()
    return fig


def plot_estimator_2d(
    estimator: "Estimator",
    x_samples: "NMatrix",
    xp_samples: "NMatrix",
    X_bounds: "RectSet | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    show: bool = True,
) -> go.Figure:
    """Plot the estimator's predictions against the true system dynamics in 2D."""
    fig = go.Figure()

    if X_bounds is not None:
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[X_bounds.lower_bound[0], X_bounds.upper_bound[0]]),
                yaxis=dict(range=[X_bounds.lower_bound[1], X_bounds.upper_bound[1]]),
            )
        )

    if X_init is not None:
        fig = plot_set_2d(X_init, color="green", label="Initial Set", fig=fig)
    if X_unsafe is not None:
        fig = plot_set_2d(X_unsafe, color="red", label="Unsafe Set", fig=fig)

    xp_pred = estimator.predict(x_samples)

    # Plot the true vs predicted next states
    fig.add_trace(
        go.Scatter3d(
            x=x_samples[:, 0],
            y=x_samples[:, 1],
            z=xp_samples[:, 0],
            mode="markers",
            marker=dict(color="blue", symbol="circle"),
            name="Ground truth",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=x_samples[:, 0],
            y=x_samples[:, 1],
            z=xp_pred[:, 0],
            mode="markers",
            marker=dict(color="orange", symbol="x"),
            name="Estimator prediction",
        )
    )

    fig.update_layout(
        title="Estimator Predictions vs True Dynamics",
        scene=dict(xaxis_title="State x[0]", yaxis_title="State x[1]", zaxis_title="Next State"),
    )

    if show:
        fig.show()
    return fig


def plot_feature_map(
    feature_map: "FeatureMap",
    x_samples: "NMatrix | Callable[[], NMatrix] | None" = None,
    X_bounds: "Set | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    num_plots: int = 10,
    N: int = 300,
    show: bool = True,
) -> go.Figure:
    """Plot the feature map of the system dynamics."""
    assert feature_map is not None, "Feature map must be provided."
    assert N > 0, "N must be a positive integer."

    fig = go.Figure()

    # If we are given a set, we get a lattice of points
    if callable(x_samples):
        x_samples = x_samples()
    elif x_samples is None:
        x_samples = X_bounds.lattice(N, True)

    for i in range(num_plots):
        B_x = feature_map(x_samples)
        sol = np.random.rand(B_x.shape[1]) * 20 - 10
        val = B_x @ sol.T

        fig.add_trace(
            go.Scatter(x=x_samples.flatten(), y=val.flatten(), mode="lines", name=f"Feature {i+1}", showlegend=False)
        )

    if X_bounds is not None:
        fig.update_xaxes(range=[X_bounds.lower_bound[0], X_bounds.upper_bound[0]])
    if X_init is not None:
        fig = plot_set_1d(X_init, "blue", label="Initial Set", fig=fig)
    if X_unsafe is not None:
        fig = plot_set_1d(X_unsafe, "red", label="Unsafe Set", fig=fig)

    fig.update_layout(title="Feature Map Visualization", xaxis_title="State space", yaxis_title="Feature Map Value")

    if show:
        fig.show()
    return fig


def plot_estimator(
    estimator: "Estimator",
    x_samples: "NMatrix | Callable[[], NMatrix] | None",
    xp_samples: "NMatrix | Callable[[NMatrix], NMatrix]",
    X_bounds: "RectSet | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    N: int = 300,
    show: bool = True,
) -> go.Figure:
    """Plot the estimator's predictions against the true system dynamics."""
    assert estimator is not None, "Estimator must be provided."
    assert xp_samples is not None, "xp_samples must be provided."
    assert N > 0, "N must be a positive integer."

    # If we are given a set, we get a lattice of points
    if callable(x_samples):
        x_samples = x_samples()
    elif x_samples is None:
        x_samples = X_bounds.lattice(N, True)

    if callable(xp_samples):
        xp_samples = xp_samples(x_samples)

    assert x_samples.ndim == 2, "x_samples must be a 2D array (num_samples x num_dimensions)."
    assert xp_samples.ndim == 2, "xp_samples must be a 2D array (num_samples x num_dimensions)."

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
    num_samples: "int | None" = None,
    show: bool = True,
) -> go.Figure:
    assert X_bounds.dimension > 0, "X_bounds must have a positive dimension."
    plot_solution_fun = (plot_solution_1d, plot_solution_2d)
    if X_bounds.dimension <= len(plot_solution_fun):
        return plot_solution_fun[X_bounds.dimension - 1](
            X_bounds, X_init, X_unsafe, feature_map, sol, eta, gamma, estimator, f, c, num_samples, show
        )
    raise exception.LucidNotSupportedException(
        f"Plotting is not supported for {X_bounds.dimension}-dimensional sets. Only 1D and 2D are supported."
    )


def plot_function_1d(
    X_bounds: "RectSet",
    f: "Callable[[NMatrix], NMatrix] | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    x_samples: "NMatrix" = np.empty((0, 0), dtype=np.float64),
    xp_samples: "NMatrix" = np.empty((0, 0), dtype=np.float64),
    n: int = 100,
    show: bool = True,
) -> go.Figure:
    """Plot a function f over the given samples in 1D."""
    assert X_bounds.dimension == 1, "plot_function is only supported for 1D functions."

    fig = go.Figure()

    if X_init is not None:
        fig = plot_set_1d(X_init, "blue", label="Initial Set", fig=fig)
    if X_unsafe is not None:
        fig = plot_set_1d(X_unsafe, "red", label="Unsafe Set", fig=fig)

    if len(x_samples) == 0:
        x_samples = X_bounds.lattice(n, True).flatten()
    n = len(x_samples)
    if len(xp_samples) == 0:
        assert_or_raise(f is not None, "Function f must be provided if xp_samples is not given.")
        xp_samples = f(x_samples.reshape(-1, 1)).flatten()
    assert xp_samples.ndim == 1 or xp_samples.shape[1] == 1, "Function f must return a 1D array for 1D plotting."

    y = np.linspace(0, 1, n)

    u = np.repeat((xp_samples - x_samples).reshape(1, -1), n, axis=0)
    v = np.zeros((n, n))  # Assuming a 1D function, v is zero

    fig = ff.create_streamline(
        x_samples,
        y,
        u,
        v,
        arrow_scale=0.05,
        density=0.2,
    )
    fig.update_layout(title="Function Plot", xaxis_title="Input Dimension")
    if show:
        fig.show()
    return fig


def plot_function_2d(
    X_bounds: "RectSet",
    f: "Callable[[NMatrix], NMatrix] | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    x_samples: "NMatrix" = np.empty((0, 0), dtype=np.float64),
    xp_samples: "NMatrix" = np.empty((0, 0), dtype=np.float64),
    n: int = 100,
    show: bool = True,
) -> go.Figure:
    """Plot a function f over the given samples in 2D."""
    assert X_bounds.dimension == 2, "plot_function is only supported for 2D functions."

    if len(x_samples) == 0:
        x_samples = X_bounds.lattice(n, True)
    n = int(np.sqrt(len(x_samples)))
    X = x_samples[:, 0].reshape(n, n)
    Y = x_samples[:, 1].reshape(n, n)

    if len(xp_samples) == 0:
        assert_or_raise(f is not None, "Function f must be provided if xp_samples is not given.")
        xp_samples = f(x_samples)

    assert xp_samples.ndim == 2 and xp_samples.shape[1] == 2, "Function f must return a 2D array for 2D plotting."

    Xp = xp_samples[:, 0].reshape(n, n)
    Yp = xp_samples[:, 1].reshape(n, n)

    u = Xp - X
    v = Yp - Y

    speed = np.sqrt(u**2 + v**2)
    speed[speed == 0] = 1

    fig = ff.create_streamline(X[0, :], Y[:, 0], u, v, density=2)
    if X_init is not None:
        fig = plot_set_2d_plane(X_init, "blue", label="Initial Set", fig=fig)
    if X_unsafe is not None:
        fig = plot_set_2d_plane(X_unsafe, "red", label="Unsafe Set", fig=fig)
    fig.update_layout(title="Function Plot", xaxis_title="Input Dimension 1", yaxis_title="Input Dimension 2")

    if show:
        fig.show()
    return fig


def plot_function(
    X_bounds: "RectSet",
    f: "Callable[[NMatrix], NMatrix] | None" = None,
    X_init: "Set | None" = None,
    X_unsafe: "Set | None" = None,
    x_samples: "NMatrix" = np.empty((0, 0), dtype=np.float64),
    xp_samples: "NMatrix" = np.empty((0, 0), dtype=np.float64),
    n: int = 100,
    show: bool = True,
) -> go.Figure:
    """Plot a function f over the given samples."""
    plot_function_fun = (plot_function_1d, plot_function_2d)
    if X_bounds.dimension <= len(plot_function_fun):
        return plot_function_fun[X_bounds.dimension - 1](
            X_bounds=X_bounds,
            f=f,
            X_init=X_init,
            X_unsafe=X_unsafe,
            x_samples=x_samples,
            xp_samples=xp_samples,
            n=n,
            show=show,
        )
    raise exception.LucidNotSupportedException(
        f"Plotting is not supported for {X_bounds.dimension}-dimensional sets. Only 1D and 2D are supported."
    )
