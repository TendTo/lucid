import numpy as np
import pytest

from pylucid import (
    AlglibOptimiser,
    GaussianKernel,
    GridSearchTuner,
    KernelRidgeRegressor,
    LinearTruncatedFourierFeatureMap,
    MultiSet,
    Parameter,
    ParameterValues,
    RectSet,
    Stats,
)

SIGMA_L = np.ones((4,))
X = np.array([[1.0, 2.0, 3.0, 5.0], [4.0, 5.0, 6.0, 1.0]])
Y = np.array([[1.0, 2.0], [5.0, 6.0]])


class TestStats:
    def test_init(self):
        with Stats() as stats:
            assert isinstance(stats, Stats)
            assert stats is not None
            assert stats.estimator_time == 0.0
            assert stats.feature_map_time == 0.0
            assert stats.optimiser_time == 0.0
            assert stats.tuning_time == 0.0
            assert stats.kernel_time == 0.0
            assert stats.total_time > 0
            assert stats.num_estimator_consolidations == 0
            assert stats.num_kernel_applications == 0
            assert stats.num_feature_map_applications == 0
            assert stats.num_tuning == 0
            assert stats.num_constraints == 0
            assert stats.num_variables == 0
            assert stats.peak_rss_memory_usage == 0
            assert str(stats).startswith("Stats:")
            stats.collect_peak_rss_memory_usage()
            assert stats.peak_rss_memory_usage > 0

    def test_no_context(self):
        stats = Stats()
        assert stats is not None
        with pytest.raises(RuntimeError):
            _ = stats.estimator_time
        with pytest.raises(RuntimeError):
            _ = stats.total_time
        with pytest.raises(RuntimeError):
            _ = stats.num_estimator_consolidations
        with pytest.raises(RuntimeError):
            _ = stats.num_kernel_applications
        with pytest.raises(RuntimeError):
            _ = stats.num_feature_map_applications
        with pytest.raises(RuntimeError):
            _ = stats.num_tuning
        with pytest.raises(RuntimeError):
            _ = stats.num_constraints
        with pytest.raises(RuntimeError):
            _ = stats.num_variables
        with pytest.raises(RuntimeError):
            _ = stats.peak_rss_memory_usage

    def test_print_no_context(self):
        stats = Stats()
        assert stats is not None
        assert str(stats) == "No stats available. Make sure the object is within the 'with' block it was defined in"

    def test_estimator_stats(self):
        with Stats() as stats:
            o = KernelRidgeRegressor(kernel=GaussianKernel(sigma_l=SIGMA_L))
            o.consolidate(x=X, y=Y)
            estimator_time = stats.estimator_time
            assert estimator_time > 0
            assert stats.num_estimator_consolidations == 1
            o.consolidate(x=X, y=Y)
            assert stats.num_estimator_consolidations == 2
            assert stats.estimator_time > estimator_time
            assert str(stats).startswith("Stats:")

    def test_feature_map_stats(self):
        with Stats() as stats:
            feature_map = LinearTruncatedFourierFeatureMap(4, np.array([1.0, 1.0]), 1.0, RectSet([-1, -1], [1, 1]))
            feature_map(Y)
            feature_map_time = stats.feature_map_time
            assert feature_map_time > 0
            assert stats.num_feature_map_applications == 1
            feature_map(Y)
            assert stats.num_feature_map_applications == 2
            assert stats.feature_map_time > feature_map_time
            assert str(stats).startswith("Stats:")

    def test_tuning_stats(self):
        with Stats() as stats:
            params = [
                ParameterValues(Parameter.SIGMA_L, [np.array([0.1, 0.1, 0.1, 0.1]), np.array([1.0, 1.0, 1.0, 1.0])]),
                ParameterValues(Parameter.SIGMA_F, [0.1, 1.0]),
            ]
            o = KernelRidgeRegressor(kernel=GaussianKernel(sigma_l=SIGMA_L), tuner=GridSearchTuner(params))
            o.fit(X, Y)
            tuner_time = stats.tuning_time
            assert tuner_time > 0
            assert stats.num_tuning == 1
            o.fit(X, Y)
            assert stats.num_tuning == 2
            assert stats.tuning_time > tuner_time
            assert str(stats).startswith("Stats:")

    def test_optimiser_stats(self):
        with Stats() as stats:
            n_per_dim = 32
            sigma_f = 15.0
            X_bounds = RectSet([(-1, 1)])
            X_init = RectSet([(-0.5, 0.5)])
            X_unsafe = MultiSet(RectSet([(-1, -0.9)]), RectSet([(0.9, 1)]))

            feature_map = LinearTruncatedFourierFeatureMap(4, 1.75555556, sigma_f, X_bounds)

            x_lattice = X_bounds.lattice(n_per_dim, True)
            u_f_x_lattice = feature_map(x_lattice)
            u_f_xp_lattice = feature_map(x_lattice / 2.0)

            x0_lattice = X_init.lattice(n_per_dim, True)
            f_x0_lattice = feature_map(x0_lattice)

            xu_lattice = X_unsafe.lattice(n_per_dim, True)
            f_xu_lattice = feature_map(xu_lattice)

            o = AlglibOptimiser(5, 1.0, 0, 1.0, 1.0, sigma_f)
            o.solve(
                f0_lattice=f_x0_lattice,
                fu_lattice=f_xu_lattice,
                phi_mat=u_f_x_lattice,
                w_mat=u_f_xp_lattice,
                rkhs_dim=feature_map.dimension,
                num_frequencies_per_dim=4 - 1,
                num_frequency_samples_per_dim=n_per_dim,
                original_dim=X_bounds.dimension,
                callback=lambda *args, **kwargs: None,
            )

            assert stats.num_constraints == 321
            assert stats.num_variables == 13
            assert stats.optimiser_time > 0
            assert str(stats).startswith("Stats:")
