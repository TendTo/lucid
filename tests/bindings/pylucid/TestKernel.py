from pylucid import GaussianKernel, Kernel
import numpy as np

class TestKernel:
    class TestGaussianKernel:

        def test_init_params(self):
            k = GaussianKernel(params=[1, 2, 3, 4])
            assert np.allclose(k.parameters, [1, 2, 3, 4])
            assert k.sigma_f == 1
            assert np.allclose(k.sigma_l, [2.0, 3.0, 4.0])

        def test_init_sigmas(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            assert np.allclose(k.parameters, [2.0, 3.0, 4.0, 5.0])
            assert k.sigma_f == 2
            assert np.allclose(k.sigma_l, [3.0, 4.0, 5.0])

        def test_call(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            assert k([1, 2, 3], [1, 2, 3]) == 4

        def test_clone(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            kc = k.clone()
            assert kc is not k
            assert np.allclose(kc.parameters, k.parameters)

        def test_clone_params(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            kc = k.clone([6, 7, 8, 9])
            assert kc is not k
            assert np.allclose(kc.parameters, [6.0, 7.0, 8.0, 9.0])
