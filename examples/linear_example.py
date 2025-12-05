import numpy as np

from pylucid import *


def main():
    config = Configuration(seed=42, verbose=log.LOG_DEBUG)

    log.set_verbosity(config.verbose)
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Define system dynamics to generate the training data
    def system_dynamics(x):
        return x / 2

    def noisy_system_dynamics(x):
        return system_dynamics(x) + np.random.normal(scale=0.01)

    # Define sets
    X_bounds = RectSet([-1.0], [1.0])
    X_init = RectSet([-0.5], [0.5])
    X_unsafe = MultiSet(RectSet([-1.0], [-0.9]), RectSet([0.9], [1.0]))

    # Define feature map
    feature_map = LinearTruncatedFourierFeatureMap(
        num_frequencies=6,
        sigma_f=1.0,
        sigma_l=np.array([0.0925]),
        X_bounds=X_bounds,
    )

    # Generate training samples
    num_samples = 1000
    x_samples = X_bounds.sample(num_samples)
    xp_samples = noisy_system_dynamics(x_samples)
    fxp_samples = feature_map(xp_samples)

    # Fit the model (A tuning procedure is recommended)
    estimator = KernelRidgeRegressor(
        kernel=GaussianKernel(sigma_l=0.04465750366442458, sigma_f=1.0),
        regularization_constant=1e-5,
    )
    estimator.fit(x_samples, fxp_samples)
    x_val = X_bounds.sample(100)
    log.info(f"Model score: {estimator.score(x_val, feature_map(system_dynamics(x_val)))}")

    # Synthesize the barrier certificate
    b = FourierBarrierCertificate(T=15, gamma=1.0)
    success = b.synthesize(
        X_bounds=X_bounds,
        X_init=X_init,
        X_unsafe=X_unsafe,
        estimator=estimator,
        feature_map=feature_map,
        lattice_resolution=300,
        parameters=FourierBarrierCertificateParameters(
            set_scaling=0.04,
        ),
    )
    if success:
        log.info(f"Barrier certificate synthesis succeeded with safety {b.safety}")
    else:
        log.error("Barrier certificate synthesis failed")


if __name__ == "__main__":
    main()
