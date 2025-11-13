from pylucid import *
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

matplotlib.use("WebAgg")

def feature_map_on_rectset(f: LinearTruncatedFourierFeatureMap, X_bounds: RectSet, n_samples: int = 100):
    x = np.linspace(X_bounds.lower_bound[0], X_bounds.upper_bound[0], n_samples)
    y = np.linspace(X_bounds.lower_bound[1], X_bounds.upper_bound[1], n_samples)
    X, Y = np.meshgrid(x, y)
    pos = np.vstack([X.ravel(), Y.ravel()]).T
    Z = f(pos)
    return X, Y, Z


def main():
    num_freq = 2
    sigma_f = 1.0
    sigma_l = np.array([0.1, 2.0])
    X_bounds = RectSet([[-3.0, 3.0], [-3.0, 3.0]])
    plot_cols = 3

    f = LinearTruncatedFourierFeatureMap(num_frequencies=num_freq, sigma_f=sigma_f, sigma_l=sigma_l, X_bounds=X_bounds)

    # plot the feature map for each f.dimension such that there are at most 3 columns
    # Create 3D subplots for surface plotting
    fig, axes = plt.subplots(
        nrows=(f.dimension + plot_cols - 1) // plot_cols,
        ncols=min(f.dimension, plot_cols),
        figsize=(12, 8),
        subplot_kw={"projection": "3d"},
    )

    n_samples = 100
    X, Y, Z = feature_map_on_rectset(f, X_bounds, n_samples)
    X_p, Y_p, Z_p = feature_map_on_rectset(f, f.get_periodic_set(), n_samples)

    # Ensure axes is iterable (handle single-axis case)
    axes_list = axes.flatten() if hasattr(axes, "flatten") else np.array([axes])

    for i, ax in zip(range(f.dimension), axes_list):
        Z_i = Z[:, i].reshape(X.shape)
        Z_p_i = Z_p[:, i].reshape(X_p.shape)
        # Plot 3D surface
        ax.plot_wireframe(X_p, Y_p, Z_p_i, color='gray', linewidth=0.5, alpha=0.5)
        ax.plot_surface(X, Y, Z_i, cmap="viridis", linewidth=0, antialiased=True)

        ax.set_title(f"Feature {i+1}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.view_init(elev=30, azim=-60)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
