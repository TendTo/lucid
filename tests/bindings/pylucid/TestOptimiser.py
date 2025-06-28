from pylucid import GurobiOptimiser


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
