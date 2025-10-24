"""PSO test / diagnostic script for the Vallée–Poussin objective.

This example builds a periodic, equidistant lattice on a [0, 2*pi)^n domain,
marks an initial rectangular set (``X_init``) inside that domain and treats the
remaining lattice points (``x0_lattice_wo_init``) as the excluded set used to
build a Vallée–Poussin (VP) remainder objective.

Features
- Periodic lattice creation (endpoint excluded) and shortest-distance wrapping
    of coordinate differences (correct angular distances for the VP kernel).
- Construction of the VP objective: sum_j D(x - x_j) / Ntilde where D is the
    Valée–Poussin kernel evaluated on wrapped differences and Ntilde is the total
    number of lattice points.
- Minimal diagnostics: FIELD (objective) max/mean, the multidimensional
    theoretical bound ((sqrt(Q/(Q-2n)))^d) and the sampled sup_x of the
    normalized absolute-basis-sum over the full lattice (the quantity appearing
    in the paper's theorem).
- PSO optimisation (pyswarms GlobalBestPSO) with the ability to visualise
    the field, particles and 3D surface.

Dependencies
- numpy, matplotlib, pyswarms
- Optional: scipy for local refinement (not required by the example)

This script is intended for exploration, diagnostics and visualisation; it
contains compact helpers (``plotting``, ``diagnostics``, ``rescale_sets``)
that can be extracted into a test harness or reused in pipelines.
"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

try:
    import pyswarms as ps
except ImportError:
    raise ImportError("pyswarms is required to run this example (pip install pyswarms)")


def vallee_poussin_kernel(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """Vectorised Vallée–Poussin kernel.

    z: array (N, dim)
    returns array (N,)
    """
    z = np.atleast_2d(z)
    N, dim = z.shape
    coeff = 1.0 / ((b - a) ** dim)
    prod = np.ones(N)
    for i in range(dim):
        zi = z[:, i]
        numerator = np.sin(((b + a) / 2) * zi) * np.sin(((b - a) / 2) * zi)
        denominator = np.sin(zi / 2) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(denominator != 0, numerator / denominator, (b ** 2 - a ** 2))
        prod *= fraction
    return coeff * prod

def build_lattice(lb, ub, per_dim):
    # per_dim: number of lattice points per dimension (scalar or 2-tuple)
    if np.isscalar(per_dim):
        per_dim = (int(per_dim),) * len(lb)
    # For periodic lattices (e.g. [0, 2*pi)), exclude the upper endpoint so we get
    # exactly `n` equidistant points per dimension (endpoint=False).
    grids = [np.linspace(l, u, n, endpoint=False) for l, u, n in zip(lb, ub, per_dim)]
    mesh = np.meshgrid(*grids, indexing="xy")
    pts = np.vstack([m.ravel() for m in mesh]).T
    return pts

def _wrap_periodic_diffs(diffs, period=2.0 * np.pi):
    """Wrap differences to the principal interval [-period/2, period/2].

    diffs: array with last axis = dim, can be shape (P, Q, dim) or (N, dim)
    Returns wrapped differences with same shape.
    """
    # ensure ndarray
    d = np.asarray(diffs)
    # shift to [-period/2, period/2] via modulo
    half = period / 2.0
    # Use vectorised modulo: ((d + half) % period) - half
    return (np.mod(d + half, period) - half)

def plotting(XX, YY, FIELD, lattice, x0_lattice_wo_init, optimizer, best_pos, init_lb, init_ub, init_lb_rescale, init_ub_rescale, percent, f_max, Q_tilde, Ntilde):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 6))

    # nicer colormap and contour lines for clarity
    im = ax1.contourf(XX, YY, FIELD, levels=80, cmap="plasma")
    cs = ax1.contour(XX, YY, FIELD, levels=10, colors="white", linewidths=0.6, alpha=0.8)
    ax1.clabel(cs, fmt="%.3f", fontsize=8)
    cbar = fig.colorbar(im, ax=ax1)
    cbar.set_label("VP objective")

    # lattice (background) as light gray plus signs
    ax1.scatter(lattice[:, 0], lattice[:, 1], s=40, marker="+", color="#cccccc", linewidths=0.5, zorder=1, label="lattice on X0")
    # excluded points as black plus signs
    ax1.scatter(x0_lattice_wo_init[:, 0], x0_lattice_wo_init[:, 1], s=40, marker="+", color="k", linewidths=0.5, zorder=3, label="lattice not on X0")

    # show the initial rectangle
    from matplotlib.patches import Rectangle

    rect = Rectangle((init_lb[0], init_lb[1]), init_ub[0] - init_lb[0], init_ub[1] - init_lb[1],
                     linewidth=2, edgecolor="#ff00dd", facecolor="none", zorder=6, linestyle="--")
    ax1.add_patch(rect)
    if percent > 0:
        rect2 = Rectangle((init_lb_rescale[0], init_lb_rescale[1]),
                          init_ub_rescale[0] - init_lb_rescale[0],
                          init_ub_rescale[1] - init_lb_rescale[1],
                          linewidth=2, edgecolor="#00ff00", facecolor="none", zorder=6, linestyle="-.")
        ax1.add_patch(rect2)

    # final particle positions with visible edge
    particles = optimizer.swarm.position
    ax1.scatter(particles[:, 0], particles[:, 1], s=20, facecolors="#3cff00", edgecolors="k", linewidths=0.6,
                alpha=0.95, zorder=5, label="particles")

    # best solution
    ax1.scatter(best_pos[0], best_pos[1], s=260, color="#ff00dd", marker="*", edgecolors="k", linewidths=1.0,
                zorder=6, label="best")
    ax1.annotate(f"best: ({best_pos[0]:.3f}, {best_pos[1]:.3f})", xy=(best_pos[0], best_pos[1]),
                 xytext=(best_pos[0] + 0.05, best_pos[1] + 0.05), color="#ff00dd", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"), zorder=7)

    ax1.set_title("Vallee-Poussin field and PSO result")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_aspect("equal")
    ax1.legend(loc="best")

    # focused view on initial rectangle (rescaled if percent > 0)
    # nicer colormap and contour lines for clarity
    ax2.set_xlim(init_lb_rescale[0] - 0.1, init_ub_rescale[0] + 0.1)
    ax2.set_ylim(init_lb_rescale[1] - 0.1, init_ub_rescale[1] + 0.1)
    im = ax2.contourf(XX, YY, FIELD, levels=80, cmap="plasma")
    cs = ax2.contour(XX, YY, FIELD, levels=10, colors="white", linewidths=0.6, alpha=0.8)
    ax2.clabel(cs, fmt="%.3f", fontsize=8)
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label("VP objective")

    # lattice (background) as light gray plus signs
    ax2.scatter(lattice[:, 0], lattice[:, 1], s=40, marker="+", color="#cccccc", linewidths=0.8, zorder=1, label="lattice on X0")
    # excluded points as black plus signs
    ax2.scatter(x0_lattice_wo_init[:, 0], x0_lattice_wo_init[:, 1], s=40, marker="+", color="k", linewidths=1.2, zorder=3, label="lattice not on X0")

    # show the initial rectangle
    from matplotlib.patches import Rectangle

    rect = Rectangle((init_lb[0], init_lb[1]), init_ub[0] - init_lb[0], init_ub[1] - init_lb[1],
                     linewidth=2, edgecolor="#ff00dd", facecolor="none", zorder=4, linestyle="--")
    ax2.add_patch(rect)

    if percent > 0:
        rect2 = Rectangle((init_lb_rescale[0], init_lb_rescale[1]),
                            init_ub_rescale[0] - init_lb_rescale[0],
                            init_ub_rescale[1] - init_lb_rescale[1],
                            linewidth=2, edgecolor="#00ff00", facecolor="none", zorder=6, linestyle="-.")
        ax2.add_patch(rect2)

    # final particle positions with visible edge
    particles = optimizer.swarm.position
    ax2.scatter(particles[:, 0], particles[:, 1], s=20, facecolors="#3cff00", edgecolors="k", linewidths=0.6,
                alpha=0.95, zorder=5, label="particles")

    # best solution
    ax2.scatter(best_pos[0], best_pos[1], s=260, color="#ff00dd", marker="*", edgecolors="k", linewidths=1.0,
                zorder=6, label="best")
    ax2.annotate(f"best: ({best_pos[0]:.3f}, {best_pos[1]:.3f})", xy=(best_pos[0], best_pos[1]),
                 xytext=(best_pos[0] + 0.05, best_pos[1] + 0.05), color="#ff00dd", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"), zorder=7)

    ax2.set_title("Vallee-Poussin field and PSO result")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")
    ax2.set_aspect("equal")
    ax2.legend(loc="best")

    # plot cost history (negated back to objective) with smoothing
    costs = -np.array(optimizer.cost_history)
    iters = np.arange(len(costs))
    ax3.plot(iters, costs, marker="o", markersize=4, color="#1f77b4", linewidth=1.2)
    ax3.set_title("Objective history (PSO)")
    ax3.set_xlabel("iteration")
    ax3.set_ylabel("objective")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # --- 3D surface plot for additional clarity ---
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig3 = plt.figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111, projection="3d")
        # plot surface (use a downsampled grid for speed)
        surf = ax3.plot_surface(XX, YY, FIELD, rstride=3, cstride=3, cmap="plasma", linewidth=0, antialiased=True, alpha=0.9)
        fig3.colorbar(surf, ax=ax3, shrink=0.6, aspect=10)

        # overlay contour projections (level sets) onto a plane beneath the surface
        try:
            zmin = float(np.nanmin(FIELD))
            zmax = float(np.nanmax(FIELD))
            z_offset = zmin - 0.08 * (zmax - zmin)
            levels = np.linspace(zmin, zmax, 12)
            # contour projected onto the z_offset plane for readability
            cset = ax3.contour(XX, YY, FIELD, levels=levels, zdir='z', offset=z_offset, cmap='Greys', linewidths=0.8)
            # optionally draw a light wireframe on the surface to help perceive level changes
            ax3.plot_wireframe(XX, YY, FIELD, rstride=8, cstride=8, color='k', alpha=0.12)
        except Exception:
            # if contouring fails, continue without it
            pass

        # compute objective values for particles and best_pos
        def obj_value_at(points):
            p = np.atleast_2d(points)
            diffs = _wrap_periodic_diffs(p[:, None, :] - x0_lattice_wo_init[None, :, :])
            diffs_r = diffs.reshape(-1, diffs.shape[-1])
            vals = vallee_poussin_kernel(diffs_r, f_max, Q_tilde - f_max)
            vals = vals.reshape(p.shape[0], x0_lattice_wo_init.shape[0])
            sums = vals.sum(axis=1)
            return (1.0 / Ntilde) * sums

        part_vals = obj_value_at(particles)
        best_val = obj_value_at(best_pos.reshape(1, -1))[0]

        ax3.scatter(particles[:, 0], particles[:, 1], part_vals, s=60, color="#ffee58", edgecolor="k", zorder=10)
        ax3.scatter([best_pos[0]], [best_pos[1]], [best_val], s=180, color="#d62728", marker="*", edgecolor="k", zorder=11)
        ax3.set_xlabel("x1")
        ax3.set_ylabel("x2")
        ax3.set_zlabel("objective")
        ax3.set_title("Vallee-Poussin objective surface with particles and optimum")
        ax3.view_init(elev=30, azim=220)
        plt.tight_layout()
        plt.show()
    except Exception:
        # mpl_toolkits may not be available; skip 3D plot gracefully
        print("3D plotting unavailable: skipping surface plot")

def diagnostics(FIELD, Q_tilde, f_max, n, grid_pts, lattice, Ntilde):
    # Diagnostics (kept minimal): print FIELD stats and the multidimensional theoretical bound
    # Compute 1D bound and raise to the power `n` (multidimensional extension).
    try:
        one_d_bound = np.sqrt(Q_tilde / (Q_tilde - 2.0 * f_max))
        multidim_bound = one_d_bound ** float(n)
    except Exception:
        multidim_bound = float("nan")

    print(f"Max. in set = {FIELD.max():.6g}, (mean = {FIELD.mean():.6g})")
    print(f"Theoretical bound on periodic domain = {multidim_bound:.6g}")

    # Compute the supremum over x of the normalized sum of absolute Valée–Poussin kernel
    # terms over the full lattice (this is the quantity appearing in the theorem).
    try:
        diffs_abs = _wrap_periodic_diffs(grid_pts[:, None, :] - lattice[None, :, :])
        diffs_abs_r = diffs_abs.reshape(-1, diffs_abs.shape[-1])
        vals_abs = vallee_poussin_kernel(diffs_abs_r, f_max, Q_tilde - f_max)
        vals_abs = np.abs(vals_abs)
        vals_abs = vals_abs.reshape(grid_pts.shape[0], lattice.shape[0])
        sums_abs = vals_abs.sum(axis=1)
        basis_sup = (1.0 / Ntilde) * float(np.max(sums_abs))
        print(f"Max abs-basis-sum on periodic lattice = {basis_sup:.6g}")
        if not np.isnan(multidim_bound):
            if basis_sup > multidim_bound * 1.000001:
                print("WARNING: basis abs-sum exceeds theoretical bound.")
            else:
                print("Basis abs-sum is within theoretical bound.")
    except Exception:
        print("Could not compute basis abs-sum diagnostic (likely memory/time constraints).")

def rescale_sets(periodic_bounds, init_bounds):
        # Compute rescale factor from original to periodic domain
        rescale = (2.0 * np.pi) / (periodic_bounds[1] - periodic_bounds[0])

        # periodic domain bounds
        lb = np.array([0, 0])
        ub = np.array([2*np.pi, 2*np.pi])

        # Compute rescaled bounds
        init_lb = (init_bounds[0] - periodic_bounds[0]) * rescale
        init_ub = (init_bounds[1] - periodic_bounds[0]) * rescale
        return (lb, ub), (init_lb, init_ub)


def main(domain_periodic, init, increase, Q_tilde=25):
    (lb, ub), (init_lb, init_ub) = rescale_sets(domain_periodic, init)
    n = lb.shape[0]  # dimensionality   

    # create a lattice (coarse) to serve as x_lattice
    lattice = build_lattice(lb, ub, per_dim=Q_tilde)

    # Increase X_init percentually (with respect to the periodic domain)
    percent = increase  # X% increase
    lengths = ub - lb
    increase = lengths * percent
    init_lb_rescale = np.maximum(lb, init_lb - increase / 2)
    init_ub_rescale = np.minimum(ub, init_ub + increase / 2)

    def contains_rect(point, lb_rect, ub_rect):
        return np.all(point >= lb_rect) and np.all(point <= ub_rect)

    x0_lattice_wo_init = np.array([p for p in lattice if not contains_rect(p, init_lb_rescale, init_ub_rescale)])

    # choose Q_tilde and f_max
    f_max = 3
    assert f_max <= 2*Q_tilde + 1, "f_max must be at least 2*Q_tilde + 1"
    Ntilde = Q_tilde ** n
    assert Ntilde == lattice.shape[0], "Ntilde must equal number of lattice points"

    # build PSO objective (we maximise objective, but pyswarms minimises -> return negative)
    def objfn(positions):
        pos = np.atleast_2d(positions)
        # wrap differences for periodic domain to shortest angular distance
        diffs = _wrap_periodic_diffs(pos[:, None, :] - x0_lattice_wo_init[None, :, :])
        diffs_r = diffs.reshape(-1, diffs.shape[-1])
        vals = vallee_poussin_kernel(diffs_r, f_max, Q_tilde - f_max)
        vals = vals.reshape(pos.shape[0], x0_lattice_wo_init.shape[0])
        sums = vals.sum(axis=1)
        result = (1.0 / Ntilde) * sums
        # pyswarms minimises; return negative to maximise
        return -result if positions.ndim == 2 else float(-result)

    # PSO bounds are lower and upper per dimension
    bounds = (init_lb, init_ub)

    # run PSO
    t0 = time()
    optimizer = ps.single.GlobalBestPSO(n_particles=40, dimensions=2, options={"c1": 0.5, "c2": 0.3, "w": 0.9}, bounds=bounds)
    best_cost, best_pos = optimizer.optimize(objfn, iters=150, verbose=False)
    t1 = time()
    print(f"PSO completed in {t1 - t0:.3f} seconds.")

    print("best_cost (negated objective):", best_cost)
    print("best_pos:", best_pos)



    # plot objective field (grid) for visualization
    grid_n = 200
    xs = np.linspace(lb[0], ub[0], grid_n)
    ys = np.linspace(lb[1], ub[1], grid_n)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([XX.ravel(), YY.ravel()])
    # wrap grid-to-lattice differences for periodic domain
    diffs = _wrap_periodic_diffs(grid_pts[:, None, :] - x0_lattice_wo_init[None, :, :])
    diffs_r = diffs.reshape(-1, diffs.shape[-1])
    vals = vallee_poussin_kernel(diffs_r, f_max, Q_tilde - f_max)
    vals = vals.reshape(grid_pts.shape[0], x0_lattice_wo_init.shape[0])
    sums = vals.sum(axis=1)
    field = (1.0 / Ntilde) * sums
    FIELD = field.reshape(grid_n, grid_n)

    # Compute diagnostics
    diagnostics(FIELD, Q_tilde, f_max, n, grid_pts, lattice, Ntilde)

    # Plotting
    plotting(XX, YY, FIELD, lattice, x0_lattice_wo_init, optimizer, best_pos, init_lb, init_ub, init_lb_rescale, init_ub_rescale, percent, f_max, Q_tilde, Ntilde)


if __name__ == "__main__":
    # Original periodic domain bounds
    lb_orig = np.array([-1.0, -1.0])
    ub_orig = np.array([1.0, 1.0])
    domain_periodic = (lb_orig, ub_orig)

    # Original X_init bounds
    init_lb_orig = np.array([-0.5, -0.5])
    init_ub_orig = np.array([0.5, 0.5])
    init = (init_lb_orig, init_ub_orig)
    increase = 0.1  # 10% increase

    Q_tilde = 25 # number of lattice points on periodic domain per dimension

    main(domain_periodic, init, increase=increase, Q_tilde=Q_tilde)
