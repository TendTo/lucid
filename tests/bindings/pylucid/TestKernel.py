from pylucid import GaussianKernel, Kernel


class TestKernel:
    class TestGaussianKernel:

        def test_init_params(self):
            k = GaussianKernel(params=[1, 2, 3, 4])
            assert (k.parameters == [1, 2, 3, 4]).all()
            assert k.sigma_f == 1
            assert (k.sigma_l == [2, 3, 4]).all()

        def test_init_sigmas(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            assert (k.parameters == [2, 3, 4, 5]).all()
            assert k.sigma_f == 2
            assert (k.sigma_l == [3, 4, 5]).all()

        def test_call(self):
            k = GaussianKernel(sigma_f=2, sigma_l=[3, 4, 5])
            assert k([1, 2, 3], [1, 2, 3]) == 4
