# linear.py
from pylucid import *


def scenario_config(args: CLIArgs):
    # Model
    f_det = lambda x: 0.5 * x
    f = lambda x: f_det(x) + np.random.normal(scale=0.01)

    # Sets
    X_bounds = RectSet([(-1, 1)], seed=args.seed)
    X_init = RectSet([(-0.5, 0.5)])
    X_unsafe = MultiSet(RectSet([(-1, -0.9)]), RectSet([(0.9, 1)]))

    # Sampling
    x_samples = X_bounds.sample(args.num_samples)
    xp_samples = f(x_samples)

    # Estimator
    estimator = KernelRidgeRegressor(
        kernel=GaussianKernel(sigma_f=args.sigma_f, sigma_l=args.sigma_l),
        regularization_constant=args.lambda_,
    )

    # Lucid configuration
    return ScenarioConfig(
        x_samples=x_samples,
        xp_samples=xp_samples,
        X_bounds=X_bounds,
        X_init=X_init,
        X_unsafe=X_unsafe,
        T=args.time_horizon,
        gamma=args.gamma,
        num_freq_per_dim=args.num_frequencies,
        f_det=f_det,
        estimator=estimator,
        sigma_f=args.sigma_f,
        oversample_factor=args.oversample_factor,
    )
