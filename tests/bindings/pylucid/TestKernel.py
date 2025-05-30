from pylucid import GaussianKernel, Kernel, Parameter, set_verbosity
import numpy as np

try:
    from sklearn.gaussian_process.kernels import RBF
except ImportError:
    RBF = None


class TestKernel:
    class TestGaussianKernel:

        def test_init_sigmas(self):
            sigma_l = [1, 2, 3]
            k = GaussianKernel(sigma_f=2, sigma_l=sigma_l)
            assert isinstance(k, Kernel)
            assert k.get(Parameter.SIGMA_F) == k.sigma_f == 2
            assert np.allclose(k.get(Parameter.SIGMA_L), sigma_l)
            assert np.allclose(k.sigma_l, sigma_l)

        def test_init_dimension(self):
            sigma_l = 5
            k = GaussianKernel(dimension=4, sigma_f=2, sigma_l=sigma_l)
            assert isinstance(k, Kernel)
            assert k.get(Parameter.SIGMA_F) == k.sigma_f == 2
            assert np.allclose(k.get(Parameter.SIGMA_L), [sigma_l] * 4)
            assert np.allclose(k.sigma_l, [sigma_l] * 4)

        def test_has_parameters(self):
            k = GaussianKernel(1)
            assert k.has(Parameter.SIGMA_F) and Parameter.SIGMA_F in k
            assert k.has(Parameter.SIGMA_L) and Parameter.SIGMA_L in k

        def test_data(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[1, 2, 3])

            assert isinstance(k.sigma_f, float)

            assert isinstance(k.sigma_l, np.ndarray)
            assert k.sigma_l.flags.c_contiguous
            assert not k.sigma_l.flags.writeable
            assert not k.sigma_l.flags.owndata
            assert k.sigma_l.flags.aligned

            assert isinstance(k.get(Parameter.SIGMA_L), np.ndarray)
            assert k.get(Parameter.SIGMA_L).flags.c_contiguous
            assert not k.get(Parameter.SIGMA_L).flags.owndata
            assert not k.get(Parameter.SIGMA_L).flags.writeable

        def test_call(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3.0, 4.0, 5.0])
            assert k(np.array([[1.0, 2.0, 3.0]]), np.array([[1.0, 2.0, 3.0]])) == 4

        def test_call_baseline(self):
            if RBF is None:
                return
            k = GaussianKernel(sigma_f=1, sigma_l=[0.5, 0.5, 0.5])
            rbf = RBF(length_scale=k.sigma_l, length_scale_bounds="fixed")
            x = np.array([[1.0, 2.0, 3.0]])
            y = np.array([[4.0, 5.0, 6.0]])
            assert np.allclose(k(x, y), rbf(x, y))

        def test_call_baseline_anisotropic(self):
            if RBF is None:
                return
            k = GaussianKernel(sigma_f=1, sigma_l=[3, 4, 5])
            rbf = RBF(length_scale=k.sigma_l, length_scale_bounds="fixed")
            x = np.array([[1.0, 2.0, 3.0]])
            y = np.array([[6.0, 4.0, 1.0]])
            assert np.allclose(k(x, y), rbf(x, y))

        def test_call_single(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3.0, 4.0, 5.0])
            assert k(np.array([[1.0, 2.0, 3.0]])) == 4

        def test_call_single_baseline(self):
            if RBF is None:
                return
            k = GaussianKernel(sigma_f=1, sigma_l=[3, 4, 5])
            rbf = RBF(length_scale=k.sigma_l, length_scale_bounds="fixed")
            x = np.array([[1.0, 2.0, 3.0]])
            assert np.allclose(k(x), rbf(x))

        def test_clone(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            kc: GaussianKernel = k.clone()
            assert kc is not k
            assert kc.get(Parameter.SIGMA_F) == k.get(Parameter.SIGMA_F)
            assert kc.sigma_f == k.sigma_f
            assert np.allclose(kc.get(Parameter.SIGMA_L), k.get(Parameter.SIGMA_L))
            assert np.allclose(kc.sigma_l, k.sigma_l)

        def test_str(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            assert str(k).startswith("GaussianKernel")
