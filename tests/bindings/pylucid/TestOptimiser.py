import pytest

from pylucid import GurobiOptimiser, LinearTruncatedFourierFeatureMap, MultiSet, RectSet


class TestOptimiser:
    class TestGurobiOptimiser:

        def test_init(self):
            T = 10
            gamma = 0.1
            epsilon = 0.01
            b_norm = 0.5
            b_kappa = 0.3
            sigma_f = 0.3
            o = GurobiOptimiser(T, gamma, epsilon, b_norm, b_kappa, sigma_f)
            assert o is not None

        @pytest.mark.skip("Requires Gurobi installation")
        def test_run(self):
            T = 5
            num_frequencies = 4
            n_per_dim = 32
            sigma_f = 15.0
            sigma_l = 1.75555556
            b_norm = 1.0
            b_kappa = 1.0
            gamma = 1.0
            epsilon = 0
            X_bounds = RectSet([(-1, 1)])
            X_init = RectSet([(-0.5, 0.5)])
            X_unsafe = MultiSet(RectSet([(-1, -0.9)]), RectSet([(0.9, 1)]))

            feature_map = LinearTruncatedFourierFeatureMap(num_frequencies, sigma_l, sigma_f, X_bounds)

            x_lattice = X_bounds.lattice(n_per_dim, True)
            u_f_x_lattice = feature_map(x_lattice)
            u_f_xp_lattice = feature_map(x_lattice / 2.0)

            x0_lattice = X_init.lattice(n_per_dim, True)
            f_x0_lattice = feature_map(x0_lattice)

            xu_lattice = X_unsafe.lattice(n_per_dim, True)
            f_xu_lattice = feature_map(xu_lattice)

            o = GurobiOptimiser(T, gamma, epsilon, b_norm, b_kappa, sigma_f)
            o.solve(
                f0_lattice=f_x0_lattice,
                fu_lattice=f_xu_lattice,
                phi_mat=u_f_x_lattice,
                w_mat=u_f_xp_lattice,
                rkhs_dim=feature_map.dimension,
                num_frequencies_per_dim=num_frequencies - 1,
                num_frequency_samples_per_dim=n_per_dim,
                original_dim=X_bounds.dimension,
                callback=lambda *args, **kwargs: None,
            )
