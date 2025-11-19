import pytest

from pylucid import GurobiOptimiser, LinearTruncatedFourierFeatureMap, MultiSet, RectSet


class TestOptimiser:
    class TestGurobiOptimiser:

        def test_init(self):
            o = GurobiOptimiser()
            assert o is not None
            assert o.problem_log_file == ""
            assert o.iis_log_file == ""

        def test_init(self):
            o = GurobiOptimiser(problem_log_file="test_log.lp", iis_log_file="test_iis.ilp")
            assert o is not None
            assert o.problem_log_file == "test_log.lp"
            assert o.iis_log_file == "test_iis.ilp"

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
                problem=None,
                callback=lambda *args, **kwargs: None,
            )
