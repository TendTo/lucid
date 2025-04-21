import math

import numpy as np
from pylucid import (
    read_matrix,
    GaussianKernel,
    TruncatedFourierFeatureMap,
    RectSet,
    MultiSet,
    GaussianKernelRidgeRegression,
    project,
    GurobiLinearOptimiser,
    LucidNotSupportedException,
    GUROBI_BUILD
)


class TestIntegration:

    def test_barrier_3(self):
        ######## PARAMS ########
        tolerance = 1e-3
        num_supp_per_dim = 12
        dimension = 2
        num_freq_per_dim = 6
        sigma_f = 19.456
        sigma_l = [30, 23.568]
        b_norm = 25
        kappa_b = 1.0
        gamma = 18.312
        T = 10
        lmda = 1e-5
        N = 1000
        epsilon = 1e-3
        autonomous = True

        limit_set = RectSet((-3, -2), (2.5, 1))
        initial_set = MultiSet(
            RectSet((1, -0.5), (2, 0.5)), RectSet((-1.8, -0.1), (-1.2, 0.1)), RectSet((-1.4, -0.5), (-1.2, 0.1))
        )
        unsafe_set = MultiSet(RectSet((0.4, 0.1), (0.6, 0.5)), RectSet((0.4, 0.1), (0.8, 0.3)))

        samples_per_dim = 2 * num_freq_per_dim
        factor = math.ceil(num_supp_per_dim / samples_per_dim) + 1
        n_per_dim = factor * samples_per_dim

        x_samples = read_matrix("tests/bindings/pylucid/x_samples.matrix")
        xp_samples = read_matrix("tests/bindings/pylucid/xp_samples.matrix")

        ######## CODE ########
        k = GaussianKernel(sigma_f, sigma_l)
        assert k is not None
        tffm = TruncatedFourierFeatureMap(num_freq_per_dim, dimension, sigma_l, sigma_f, limit_set)
        assert tffm is not None

        x_lattice = limit_set.lattice(samples_per_dim)
        assert x_lattice is not None

        f_lattice = tffm(x_lattice)
        assert f_lattice.shape == (samples_per_dim ** dimension, num_freq_per_dim ** dimension * 2 - 1)
        fp_samples = tffm(xp_samples)
        assert fp_samples.shape == (xp_samples.shape[0], num_freq_per_dim ** dimension * 2 - 1)

        r = GaussianKernelRidgeRegression(k, x_samples, fp_samples, lmda)
        assert r is not None

        if_lattice = r(x_lattice)
        assert if_lattice.shape == (144, 71)

        w_mat = np.zeros((n_per_dim ** dimension, fp_samples.shape[1]))
        phi_mat = np.zeros((n_per_dim ** dimension, fp_samples.shape[1]))
        assert w_mat.shape == (576, 71)
        assert phi_mat.shape == (576, 71)
        for i in range(w_mat.shape[1]):
            w_mat[:, i] = project(if_lattice[:, i], n_per_dim, samples_per_dim, dimension)
            phi_mat[:, i] = project(f_lattice[:, i], n_per_dim, samples_per_dim, dimension)

        x0_lattice = initial_set.lattice(n_per_dim - 1, True)
        assert x0_lattice.shape == (1587, 2)
        xu_lattice = unsafe_set.lattice(n_per_dim - 1, True)
        assert xu_lattice.shape == (1058, 2)

        f0_lattice = tffm(x0_lattice)
        assert f0_lattice.shape == (1587, num_freq_per_dim ** dimension * 2 - 1)
        fu_lattice = tffm(xu_lattice)
        assert fu_lattice.shape == (1058, num_freq_per_dim ** dimension * 2 - 1)

        o = GurobiLinearOptimiser(T, gamma, epsilon, b_norm, kappa_b, sigma_f)

        def check_cb(success, obj_val, eta, c, norm):
            assert success
            assert math.isclose(obj_val, 0.8375267440200334, rel_tol=tolerance)
            assert math.isclose(eta, 15.336789736494852, rel_tol=tolerance)
            assert math.isclose(c, 0, rel_tol=tolerance)
            assert math.isclose(norm, 10.39392985811301, rel_tol=tolerance)

        try:
            assert o.solve(
                f0_lattice,
                fu_lattice,
                phi_mat,
                w_mat,
                tffm.dimension,
                num_freq_per_dim - 1,
                n_per_dim,
                dimension,
                check_cb,
            )
            assert GUROBI_BUILD
        except LucidNotSupportedException:
            assert not GUROBI_BUILD  # Did not compile against Gurobi. Ignore this test.
